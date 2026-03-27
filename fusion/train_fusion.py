"""
Fusion Model Training Pipeline
==============================
Baseline trainer for multimodal fusion model.

Supported training targets (if available in feature file):
    - affect_state: [B, 2] regression (valence, arousal)
    - stress_level: [B, 1] regression (0..1)
    - cognitive_load: [B, 1] regression (0..1)
    - engagement_score: [B, 1] regression (0..1)
    - lie_risk: [B, 1] regression (0..1 proxy)

Input feature format:
    NPZ file with keys such as:
      rppg_train, emotion_train, behavioral_train, y_train
      rppg_val, emotion_val, behavioral_val, y_val
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from fusion.fusion_model import create_fusion_model
from fusion.loso_sampler import LOSOSampler
from fusion.deception_dataset import DeceptionDataset
from fusion.real_feature_extractor import RealVideoFeatureExtractor


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalTensorDataset(Dataset):
    """Tensor dataset wrapper for multimodal sequence inputs."""

    def __init__(
        self,
        rppg: Optional[np.ndarray],
        emotion: Optional[np.ndarray],
        behavioral: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        targets: np.ndarray,
    ):
        self.inputs = {}
        if rppg is not None:
            self.inputs["rppg"] = torch.from_numpy(rppg).float()
        if emotion is not None:
            self.inputs["emotion"] = torch.from_numpy(emotion).float()
        if behavioral is not None:
            self.inputs["behavioral"] = torch.from_numpy(behavioral).float()
        if audio is not None:
            self.inputs["audio"] = torch.from_numpy(audio).float()

        self.targets = torch.from_numpy(targets).float()

        if not self.inputs:
            raise ValueError("At least one modality must be provided")

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        x = {k: v[idx] for k, v in self.inputs.items()}
        y = self.targets[idx]
        return x, y


class ManifestSequenceDataset(Dataset):
    """Dataset wrapper for prebuilt manifest sequence tensors."""

    def __init__(self, inputs: Dict[str, np.ndarray], labels: np.ndarray):
        if labels.ndim != 1:
            raise ValueError("labels must be 1D array")

        self.inputs = {k: torch.from_numpy(v).float() for k, v in inputs.items()}
        self.labels = torch.from_numpy(labels.astype(np.float32)).view(-1, 1)

        if not self.inputs:
            raise ValueError("At least one modality must be provided")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = {k: v[idx] for k, v in self.inputs.items()}
        y = self.labels[idx]
        return x, y


def _collate_fn(batch):
    """Collate batch of dict inputs with shared keys."""
    xs, ys = zip(*batch)
    keys = xs[0].keys()
    out_x = {k: torch.stack([x[k] for x in xs], dim=0) for k in keys}
    out_y = torch.stack(list(ys), dim=0)
    return out_x, out_y


def _split_targets(y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Split generic target tensor into task-specific targets.

    Expected y shape [B, >=6], order:
      0: valence
      1: arousal
      2: stress_level
      3: cognitive_load
      4: engagement_score
      5: lie_risk
    """
    if y.ndim != 2 or y.shape[1] < 6:
        raise ValueError("Targets must have shape [B, >=6]")

    return {
        "affect_state": y[:, 0:2],
        "stress_level": y[:, 2:3],
        "cognitive_load": y[:, 3:4],
        "engagement_score": y[:, 4:5],
        "lie_risk": y[:, 5:6],
    }


def compute_loss(pred: Dict[str, torch.Tensor], y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted multi-task loss."""
    target = _split_targets(y)
    mse = nn.MSELoss()

    loss_affect = mse(pred["affect_state"], target["affect_state"])
    loss_stress = mse(pred["stress_level"], target["stress_level"])
    loss_cognitive = mse(pred["cognitive_load"], target["cognitive_load"])
    loss_engage = mse(pred["engagement_score"], target["engagement_score"])
    loss_lie = mse(pred["lie_risk"], target["lie_risk"])

    # Slightly emphasize stress and lie proxy for current roadmap priority.
    total = (
        1.0 * loss_affect
        + 1.25 * loss_stress
        + 0.75 * loss_cognitive
        + 0.75 * loss_engage
        + 1.25 * loss_lie
    )

    log = {
        "loss_total": float(total.item()),
        "loss_affect": float(loss_affect.item()),
        "loss_stress": float(loss_stress.item()),
        "loss_cognitive": float(loss_cognitive.item()),
        "loss_engagement": float(loss_engage.item()),
        "loss_lie": float(loss_lie.item()),
    }
    return total, log


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss, logs = compute_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running.append(logs["loss_total"])
        pbar.set_postfix(loss=f"{logs['loss_total']:.4f}")

    return float(np.mean(running)) if running else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        pred = model(x)
        loss, _ = compute_loss(pred, y)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def weighted_bce_prob_loss(prob: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Weighted BCE on probabilities (model currently outputs sigmoid probability)."""
    prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
    target = torch.clamp(target, 0.0, 1.0)

    w_pos = float(pos_weight)
    loss = -(w_pos * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))
    return loss.mean()


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute simple binary metrics without external dependencies."""
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def train_one_epoch_deception(model, loader, optimizer, device, pos_weight: float):
    model.train()
    running = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        # Main objective: deception supervision on lie_risk head.
        loss = weighted_bce_prob_loss(pred["lie_risk"], y, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running.append(float(loss.item()))
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

    return float(np.mean(running)) if running else 0.0


@torch.no_grad()
def evaluate_deception(model, loader, device, pos_weight: float, return_predictions: bool = False):
    model.eval()
    losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    for x, y in loader:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        pred = model(x)

        prob = pred["lie_risk"]
        loss = weighted_bce_prob_loss(prob, y, pos_weight=pos_weight)
        losses.append(float(loss.item()))

        pred_cls = (prob >= 0.5).long().squeeze(-1).cpu().numpy().tolist()
        prob_vals = prob.squeeze(-1).cpu().numpy().tolist()
        true_cls = y.long().squeeze(-1).cpu().numpy().tolist()

        all_pred.extend(pred_cls)
        all_true.extend(true_cls)
        all_prob.extend([float(p) for p in prob_vals])

    metrics = compute_binary_metrics(all_true, all_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    if return_predictions:
        return metrics, {
            "y_true": [int(v) for v in all_true],
            "y_pred": [int(v) for v in all_pred],
            "y_prob": [float(v) for v in all_prob],
        }
    return metrics


def _build_model_from_dataset(dataset: MultimodalTensorDataset, cfg: TrainConfig):
    dims = {}
    for k, v in dataset.inputs.items():
        dims[k] = int(v.shape[-1])

    known = {"rppg", "emotion", "behavioral", "audio"}
    extra_dims = {k: v for k, v in dims.items() if k not in known}

    model = create_fusion_model(
        rppg_dim=dims.get("rppg"),
        emotion_dim=dims.get("emotion"),
        behavioral_dim=dims.get("behavioral"),
        audio_dim=dims.get("audio"),
        extra_dims=extra_dims,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    )
    return model


def load_manifest_loso_split(
    manifest_path: Path,
    fold_idx: int,
    modalities: List[str],
    feature_cache_dir: str,
    feature_mode: str,
    cache_version: str,
    max_video_frames: int,
    val_frac: float = 0.15,
    seed: int = 42,
    emotion_model_path: Optional[str] = None,
    temporal_augment: bool = False,
):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    extractor = None
    if feature_mode == "real":
        extractor = RealVideoFeatureExtractor(
            modalities=modalities,
            target_length=20,
            max_video_frames=max_video_frames,
            use_gesture_summary=True,
            emotion_model_path=emotion_model_path,
        )
    # Inherit any dim overrides from the extractor (e.g. CNN emotion = 7 classes)
    dim_overrides = getattr(extractor, "output_dims", {}) if extractor else {}
    dataset = DeceptionDataset(
        manifest_csv=str(manifest_path),
        feature_cache_dir=feature_cache_dir,
        modalities=modalities,
        include_gesture_features=True,
        feature_extractor=extractor,
        feature_mode=feature_mode,
        cache_version=cache_version,
        sequence_length=20,
        verbose=False,
        modality_dim_overrides=dim_overrides if dim_overrides else None,
    )

    df = dataset.manifest_df.copy()
    df["label"] = df["label"].astype(int)

    sampler = LOSOSampler(labels=df["label"].tolist(), fold_idx=fold_idx, val_frac=val_frac, seed=seed)
    split = sampler.get_fold_split()

    def take(idx_list, augment=False):
        x, y = dataset.build_tensor_pack(idx_list, modalities=modalities, target_length=20, temporal_augment=augment)
        return ManifestSequenceDataset(x, y)

    train_ds = take(split["train"], augment=temporal_augment)
    val_ds = take(split["val"]) if split["val"] else take(split["train"][:1])
    test_ds = take(split["test"])

    train_labels = df.iloc[split["train"]]["label"].to_numpy(dtype=np.int64)
    n_pos = float(np.sum(train_labels == 1))
    n_neg = float(np.sum(train_labels == 0))
    pos_weight = n_neg / max(1.0, n_pos)

    return train_ds, val_ds, test_ds, split, pos_weight


def load_npz_splits(npz_path: Path):
    """Load train/val tensors from NPZ file."""
    arr = np.load(npz_path, allow_pickle=False)

    def g(name):
        return arr[name] if name in arr else None

    train = MultimodalTensorDataset(
        rppg=g("rppg_train"),
        emotion=g("emotion_train"),
        behavioral=g("behavioral_train"),
        audio=g("audio_train"),
        targets=arr["y_train"],
    )
    val = MultimodalTensorDataset(
        rppg=g("rppg_val"),
        emotion=g("emotion_val"),
        behavioral=g("behavioral_val"),
        audio=g("audio_val"),
        targets=arr["y_val"],
    )
    return train, val


def make_synthetic_dataset(n_train=256, n_val=64, t=20):
    """Create synthetic multimodal data for smoke testing."""
    rng = np.random.default_rng(42)

    def mk(n):
        rppg = rng.normal(0, 1, size=(n, t, 8)).astype(np.float32)
        emotion = rng.normal(0, 1, size=(n, t, 16)).astype(np.float32)
        behavioral = rng.normal(0, 1, size=(n, t, 6)).astype(np.float32)
        audio = rng.normal(0, 1, size=(n, t, 6)).astype(np.float32)

        stress = (
            rppg[:, :, 0].mean(axis=1) * 0.25
            + behavioral[:, :, 1].mean(axis=1) * 0.15
            + audio[:, :, 0].mean(axis=1) * 0.10
        )
        stress = 1.0 / (1.0 + np.exp(-stress))
        valence = np.tanh(emotion[:, :, 2].mean(axis=1) * 0.5)
        arousal = np.tanh(emotion[:, :, 3].mean(axis=1) * 0.5 + stress * 0.3)
        cognitive = np.clip(stress * 0.8 + rng.normal(0, 0.05, n), 0, 1)
        engagement = np.clip(1.0 - cognitive * 0.5 + rng.normal(0, 0.05, n), 0, 1)
        lie_risk = np.clip(0.55 * stress + 0.20 * cognitive + 0.25 * (1.0 - engagement), 0, 1)

        y = np.stack([valence, arousal, stress, cognitive, engagement, lie_risk], axis=1).astype(np.float32)
        return MultimodalTensorDataset(rppg, emotion, behavioral, audio, y)

    return mk(n_train), mk(n_val)


def main():
    parser = argparse.ArgumentParser(description="Train baseline multimodal fusion model")
    parser.add_argument("--dataset", type=str, default="proxy", choices=["proxy", "reallife_2016", "boxoflies", "combined", "custom"], help="Training data source")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path for deception datasets")
    parser.add_argument("--modalities", nargs="+", default=["rppg", "emotion", "behavioral"], help="Modalities to use")
    parser.add_argument("--loso_fold", type=int, default=0, help="Leave-one-out fold index")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction from remaining samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    parser.add_argument("--feature_cache_dir", type=str, default="checkpoints/feature_cache", help="Cache directory for extracted/packed features")
    parser.add_argument("--feature_mode", type=str, default="smoke", choices=["smoke", "real"], help="Feature generation mode for manifest datasets")
    parser.add_argument("--cache_version", type=str, default="v2", help="Feature cache version tag to prevent stale cache reuse")
    parser.add_argument("--max_video_frames", type=int, default=180, help="Maximum frames to decode per video in real extraction mode")
    parser.add_argument("--features_npz", type=str, default=None, help="Path to NPZ feature split file")
    parser.add_argument("--synthetic_test", action="store_true", help="Run synthetic smoke training")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fusion")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_gesture_modality", action="store_true", help="Include manifest gesture annotations as a model modality")
    parser.add_argument("--use_audio_modality", action="store_true", help="Include MFCC audio features extracted from video as a model modality")
    parser.add_argument("--use_verbal_modality", action="store_true", help="Include transcript linguistic features as a model modality")
    parser.add_argument("--emotion_model_path", type=str, default=None, help="Path to CASME II pre-trained emotion model checkpoint (.pth)")
    parser.add_argument("--temporal_augment", action="store_true", help="Apply random temporal crop to training sequences to prevent position-bias overfitting")
    args = parser.parse_args()

    if args.use_gesture_modality and "gesture" not in args.modalities:
        args.modalities = list(args.modalities) + ["gesture"]
    if args.use_audio_modality and "audio" not in args.modalities:
        args.modalities = list(args.modalities) + ["audio"]
    if args.use_verbal_modality and "verbal" not in args.modalities:
        args.modalities = list(args.modalities) + ["verbal"]

    cfg = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_deception_mode = args.dataset != "proxy"

    if use_deception_mode:
        if not args.manifest:
            raise ValueError("--manifest is required when --dataset is not 'proxy'")
        manifest_path = Path(args.manifest)
        print(f"Loading manifest dataset from: {manifest_path}")
        train_ds, val_ds, test_ds, split, pos_weight = load_manifest_loso_split(
            manifest_path=manifest_path,
            fold_idx=args.loso_fold,
            modalities=args.modalities,
            feature_cache_dir=args.feature_cache_dir,
            feature_mode=args.feature_mode,
            cache_version=args.cache_version,
            max_video_frames=args.max_video_frames,
            val_frac=args.val_frac,
            seed=args.seed,
            emotion_model_path=getattr(args, "emotion_model_path", None),
            temporal_augment=getattr(args, "temporal_augment", False),
        )
        print(
            f"LOSO split fold={args.loso_fold} | "
            f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} | "
            f"pos_weight={pos_weight:.3f} | feature_mode={args.feature_mode}"
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    else:
        if args.synthetic_test or not args.features_npz:
            print("Using synthetic dataset for fusion smoke test...")
            train_ds, val_ds = make_synthetic_dataset()
        else:
            npz_path = Path(args.features_npz)
            if not npz_path.exists():
                raise FileNotFoundError(f"Feature file not found: {npz_path}")
            print(f"Loading feature splits from: {npz_path}")
            train_ds, val_ds = load_npz_splits(npz_path)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_fn)

    model = _build_model_from_dataset(train_ds, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val = float("inf")
    best_path = out_dir / "best_fusion_model.pth"

    print("=" * 80)
    print("Fusion training start")
    print(f"Device: {cfg.device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print("=" * 80)

    for epoch in range(1, cfg.num_epochs + 1):
        if use_deception_mode:
            train_loss = train_one_epoch_deception(model, train_loader, optimizer, cfg.device, pos_weight=pos_weight)
            val_metrics = evaluate_deception(model, val_loader, cfg.device, pos_weight=pos_weight)
            val_loss = float(val_metrics["loss"])
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
            val_loss = evaluate(model, val_loader, cfg.device)
        scheduler.step()

        if use_deception_mode:
            print(
                f"Epoch {epoch:02d}/{cfg.num_epochs} | "
                f"Train: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"ValF1: {val_metrics['f1']:.4f} | ValAcc: {val_metrics['accuracy']:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{cfg.num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val,
                "epoch": epoch,
            }, best_path)
            print(f"  New best model saved: {best_path}")

    if use_deception_mode:
        test_metrics, test_predictions = evaluate_deception(
            model,
            test_loader,
            cfg.device,
            pos_weight=pos_weight,
            return_predictions=True,
        )
        metrics_path = out_dir / "metrics.json"
        metrics_payload = {
            "dataset": args.dataset,
            "manifest": args.manifest,
            "modalities": args.modalities,
            "feature_mode": args.feature_mode,
            "fold": args.loso_fold,
            "split": {
                "train": len(split["train"]),
                "val": len(split["val"]),
                "test": len(split["test"]),
            },
            "best_val_loss": best_val,
            "test_metrics": test_metrics,
            "test_predictions": test_predictions,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        print(f"Test metrics: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")
        print(f"Metrics saved: {metrics_path}")

    print("=" * 80)
    print(f"Training completed. Best val loss: {best_val:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
