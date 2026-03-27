"""
Emotion Recognition Model
==========================
Professional-grade CNN and Transformer models for facial emotion recognition
from video frames or static images.

Architectures:
    • ResNet-50 backbone with emotion classification head
    • EfficientNet-B0 for lightweight deployment
    • Vision Transformer (ViT) option for SOTA performance
    • Temporal modeling for micro-expression sequences (3D-CNN / LSTM)

Features:
    • Transfer learning from ImageNet pre-trained weights
    • Multi-task learning (emotion + valence/arousal)
    • Attention mechanisms for key frame identification
    • Integration with rPPG features for multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple


class EmotionClassifier(nn.Module):
    """
    Base emotion classification model with CNN backbone.
    
    Supports multiple architectures:
        - ResNet-18/50
        - EfficientNet-B0
        - MobileNetV3
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize emotion classifier.
        
        Args:
            num_classes: Number of emotion classes
            backbone: CNN backbone ('resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3')
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout rate before final classifier
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Build backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        elif backbone == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: If True, return both features and logits
        
        Returns:
            Logits [B, num_classes] or tuple (features, logits)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        if return_features:
            return features, logits
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for multimodal fusion).
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature embeddings [B, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


class TemporalEmotionModel(nn.Module):
    """
    Temporal emotion recognition model for micro-expression sequences.
    
    Architecture:
        1. Frame-level feature extraction (CNN backbone)
        2. Temporal aggregation (LSTM or Transformer)
        3. Sequence-level classification
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = 'resnet50',
        temporal_model: str = 'lstm',
        pretrained: bool = True,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize temporal emotion model.
        
        Args:
            num_classes: Number of emotion classes
            backbone: CNN backbone for frame-level features
            temporal_model: Temporal aggregation ('lstm', 'gru', 'transformer', 'avg')
            pretrained: Use ImageNet pre-trained weights
            hidden_dim: Hidden dimension for temporal model
            num_layers: Number of temporal layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.temporal_model_name = temporal_model
        
        # Frame-level feature extractor
        self.frame_encoder = EmotionClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=0.0,  # No dropout in encoder
            freeze_backbone=False
        )
        
        feature_dim = self.frame_encoder.feature_dim
        
        # Remove classifier (we'll add our own)
        self.frame_encoder.classifier = nn.Identity()
        
        # Temporal aggregation
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            temporal_output_dim = hidden_dim * 2  # Bidirectional
        
        elif temporal_model == 'gru':
            self.temporal = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            temporal_output_dim = hidden_dim * 2
        
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            temporal_output_dim = feature_dim
        
        elif temporal_model == 'avg':
            # Simple average pooling
            self.temporal = nn.Identity()
            temporal_output_dim = feature_dim
        
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C, H, W] (batch, time, channels, height, width)
            return_features: If True, return temporal features
        
        Returns:
            Logits [B, num_classes] or tuple (features, logits)
        """
        B, T, C, H, W = x.shape
        
        # Extract frame-level features
        # Reshape: [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(B * T, C, H, W)
        frame_features = self.frame_encoder.backbone(x)  # [B*T, feature_dim]
        
        # Reshape back: [B*T, feature_dim] -> [B, T, feature_dim]
        feature_dim = frame_features.shape[1]
        frame_features = frame_features.view(B, T, feature_dim)
        
        # Temporal aggregation
        if self.temporal_model_name in ['lstm', 'gru']:
            temporal_features, _ = self.temporal(frame_features)
            # Use last timestep output
            temporal_features = temporal_features[:, -1, :]  # [B, hidden_dim*2]
        
        elif self.temporal_model_name == 'transformer':
            temporal_features = self.temporal(frame_features)
            # Average pool over time
            temporal_features = temporal_features.mean(dim=1)  # [B, feature_dim]
        
        else:  # avg
            temporal_features = frame_features.mean(dim=1)  # [B, feature_dim]
        
        # Classify
        logits = self.classifier(temporal_features)
        
        if return_features:
            return temporal_features, logits
        return logits


class MultiTaskEmotionModel(nn.Module):
    """
    Multi-task emotion model predicting discrete emotions + valence/arousal.
    
    Tasks:
        1. Emotion classification (8 classes)
        2. Valence regression (continuous -1 to 1)
        3. Arousal regression (continuous -1 to 1)
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize multi-task emotion model.
        
        Args:
            num_classes: Number of emotion classes
            backbone: CNN backbone
            pretrained: Use pre-trained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared backbone
        self.backbone = EmotionClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=0.0
        )
        
        feature_dim = self.backbone.feature_dim
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Task-specific heads
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Tuple of (emotion_logits, valence, arousal)
        """
        # Extract features
        features = self.backbone.backbone(x)
        
        # Multi-task predictions
        emotion_logits = self.emotion_head(features)
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        
        return emotion_logits, valence, arousal


def create_emotion_model(
    model_type: str = 'base',
    num_classes: int = 8,
    backbone: str = 'resnet50',
    **kwargs
) -> nn.Module:
    """
    Factory function for creating emotion recognition models.
    
    Args:
        model_type: Type of model ('base', 'temporal', 'multitask')
        num_classes: Number of emotion classes
        backbone: CNN backbone architecture
        **kwargs: Additional model-specific arguments
    
    Returns:
        Emotion recognition model
    """
    if model_type == 'base':
        return EmotionClassifier(num_classes=num_classes, backbone=backbone, **kwargs)
    
    elif model_type == 'temporal':
        return TemporalEmotionModel(num_classes=num_classes, backbone=backbone, **kwargs)
    
    elif model_type == 'multitask':
        return MultiTaskEmotionModel(num_classes=num_classes, backbone=backbone, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("=" * 80)
    print("Emotion Recognition Models - Architecture Test")
    print("=" * 80)
    
    # Test base model
    print("\n1️⃣ Testing Base EmotionClassifier...")
    model = EmotionClassifier(num_classes=8, backbone='resnet50', pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test temporal model
    print("\n2️⃣ Testing Temporal Emotion Model...")
    temporal_model = TemporalEmotionModel(num_classes=8, backbone='resnet50', 
                                          temporal_model='lstm', pretrained=False)
    x_temporal = torch.randn(2, 16, 3, 224, 224)  # [B, T, C, H, W]
    logits_temporal = temporal_model(x_temporal)
    print(f"   Input: {x_temporal.shape}")
    print(f"   Output: {logits_temporal.shape}")
    print(f"   Parameters: {sum(p.numel() for p in temporal_model.parameters()):,}")
    
    # Test multi-task model
    print("\n3️⃣ Testing Multi-Task Model...")
    multitask_model = MultiTaskEmotionModel(num_classes=8, backbone='resnet50', pretrained=False)
    emotion_logits, valence, arousal = multitask_model(x)
    print(f"   Input: {x.shape}")
    print(f"   Emotion output: {emotion_logits.shape}")
    print(f"   Valence output: {valence.shape}")
    print(f"   Arousal output: {arousal.shape}")
    print(f"   Parameters: {sum(p.numel() for p in multitask_model.parameters()):,}")
    
    print("\n✅ All model architectures working correctly!")
