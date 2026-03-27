"""
Multimodal Fusion Model
=======================
Baseline temporal fusion architecture for psychophysiological analysis.

Expected modality inputs (each optional):
    - rppg:       [B, T, D_rppg]
    - emotion:    [B, T, D_emotion]
    - behavioral: [B, T, D_behavioral]

Outputs:
    - affect_state: [B, 2] (valence, arousal)
    - stress_level: [B, 1] in [0, 1]
    - cognitive_load: [B, 1] in [0, 1]
    - engagement_score: [B, 1] in [0, 1]
    - lie_risk: [B, 1] in [0, 1] (proxy until deception labels are available)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Project modality-specific features to a shared hidden space."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPooling(nn.Module):
    """Attention pooling over temporal dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # [B, T]
        pooled = torch.sum(x * w.unsqueeze(-1), dim=1)       # [B, H]
        return pooled


class FusionModel(nn.Module):
    """Baseline multimodal temporal fusion model."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not input_dims:
            raise ValueError("input_dims cannot be empty")

        self.hidden_dim = hidden_dim
        self.modality_names = sorted(list(input_dims.keys()))

        self.projectors = nn.ModuleDict({
            name: ModalityProjector(dim, hidden_dim, dropout=dropout)
            for name, dim in input_dims.items()
        })

        # Lightweight transformer encoder for temporal modeling.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = AttentionPooling(hidden_dim)
        self.head_pre = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-task heads
        self.affect_head = nn.Linear(hidden_dim, 2)
        self.stress_head = nn.Linear(hidden_dim, 1)
        self.cognitive_head = nn.Linear(hidden_dim, 1)
        self.engagement_head = nn.Linear(hidden_dim, 1)
        self.lie_risk_head = nn.Linear(hidden_dim, 1)

    def _fuse_modalities(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse available modalities per timestep.

        Each tensor must be [B, T, D_modality].
        Returns fused tensor [B, T, H].
        """
        projected = []
        lengths = []

        for name in self.modality_names:
            if name not in inputs:
                continue
            x = inputs[name]
            if x.ndim != 3:
                raise ValueError(f"Input modality '{name}' must be [B, T, D], got {tuple(x.shape)}")
            projected.append(self.projectors[name](x))
            lengths.append(x.shape[1])

        if not projected:
            raise ValueError("No valid modality tensor provided to forward()")

        # Align all modalities to minimum temporal length.
        min_t = min(lengths)
        projected = [p[:, :min_t, :] for p in projected]

        # Mean fusion across modalities.
        fused = torch.stack(projected, dim=2).mean(dim=2)  # [B, T, H]
        return fused

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal sequence inputs."""
        x = self._fuse_modalities(inputs)
        x = self.temporal_encoder(x)
        x = self.pool(x)
        x = self.head_pre(x)

        out = {
            "affect_state": self.affect_head(x),
            "stress_level": torch.sigmoid(self.stress_head(x)),
            "cognitive_load": torch.sigmoid(self.cognitive_head(x)),
            "engagement_score": torch.sigmoid(self.engagement_head(x)),
            "lie_risk": torch.sigmoid(self.lie_risk_head(x)),
        }
        return out


def create_fusion_model(
    rppg_dim: Optional[int] = None,
    emotion_dim: Optional[int] = None,
    behavioral_dim: Optional[int] = None,
    audio_dim: Optional[int] = None,
    extra_dims: Optional[Dict[str, int]] = None,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.2,
) -> FusionModel:
    """Factory helper to build a fusion model with selected modalities."""
    input_dims: Dict[str, int] = {}
    if rppg_dim is not None and rppg_dim > 0:
        input_dims["rppg"] = int(rppg_dim)
    if emotion_dim is not None and emotion_dim > 0:
        input_dims["emotion"] = int(emotion_dim)
    if behavioral_dim is not None and behavioral_dim > 0:
        input_dims["behavioral"] = int(behavioral_dim)
    if audio_dim is not None and audio_dim > 0:
        input_dims["audio"] = int(audio_dim)
    if extra_dims:
        for name, dim in extra_dims.items():
            if dim is not None and int(dim) > 0:
                input_dims[str(name)] = int(dim)

    return FusionModel(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
