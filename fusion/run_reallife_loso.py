#!/usr/bin/env python3
"""Run multiple LOSO folds for Real-life deception training with resume support."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_fold(
    python_exec: str,
    manifest: str,
    cache_dir: str,
    output_root: str,
    fold: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    modalities: list[str],
    feature_mode: str,
    cache_version: str,
    max_video_frames: int,
    use_gesture_modality: bool,
    use_audio_modality: bool = False,
    use_verbal_modality: bool = False,
    hidden_dim: int = 96,
    num_layers: int = 1,
    dropout: float = 0.3,
    weight_decay: float = 5e-4,
    emotion_model_path: str = "",
    temporal_augment: bool = False,
) -> int:
    out_dir = Path(output_root) / f"fold_{fold:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exec,
        "fusion/train_fusion.py",
        "--dataset", "reallife_2016",
        "--manifest", manifest,
        "--feature_cache_dir", cache_dir,
        "--feature_mode", feature_mode,
        "--cache_version", cache_version,
        "--max_video_frames", str(max_video_frames),
        "--loso_fold", str(fold),
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--hidden_dim", str(hidden_dim),
        "--num_layers", str(num_layers),
        "--dropout", str(dropout),
        "--weight_decay", str(weight_decay),
        "--output_dir", str(out_dir),
        "--modalities",
        *modalities,
    ]
    if use_gesture_modality:
        cmd.append("--use_gesture_modality")
    if use_audio_modality:
        cmd.append("--use_audio_modality")
    if use_verbal_modality:
        cmd.append("--use_verbal_modality")
    if emotion_model_path:
        cmd.extend(["--emotion_model_path", emotion_model_path])
    if temporal_augment:
        cmd.append("--temporal_augment")

    print(f"\n=== Running fold {fold} ===")
    print(" ".join(cmd))

    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO fold range for Real-life deception training")
    parser.add_argument("--manifest", default=r"data\RealLifeDeceptionDetection.2016\deception_manifest.csv")
    parser.add_argument("--cache_dir", default="checkpoints/feature_cache")
    parser.add_argument("--output_root", default="checkpoints/deception_reallife2016_loso")
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=120)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modalities", nargs="+", default=["rppg", "emotion", "behavioral"])
    parser.add_argument("--feature_mode", choices=["smoke", "real"], default="smoke")
    parser.add_argument("--cache_version", default="v2")
    parser.add_argument("--max_video_frames", type=int, default=180)
    parser.add_argument("--use_gesture_modality", action="store_true")
    parser.add_argument("--use_audio_modality", action="store_true")
    parser.add_argument("--use_verbal_modality", action="store_true")
    parser.add_argument("--emotion_model_path", type=str, default="", help="Path to CASME II pre-trained emotion model .pth")
    parser.add_argument("--temporal_augment", action="store_true", help="Random temporal crop on training splits to prevent position-bias overfitting")
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="Skip folds that already have metrics.json")
    args = parser.parse_args()

    if args.end_fold < args.start_fold:
        raise ValueError("end_fold must be >= start_fold")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    failed = []
    for fold in range(args.start_fold, args.end_fold + 1):
        metrics_path = output_root / f"fold_{fold:03d}" / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"Skipping fold {fold}: metrics already exists")
            continue

        code = run_fold(
            python_exec=args.python_exec,
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            fold=fold,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            modalities=args.modalities,
            feature_mode=args.feature_mode,
            cache_version=args.cache_version,
            max_video_frames=args.max_video_frames,
            use_gesture_modality=args.use_gesture_modality,
            use_audio_modality=args.use_audio_modality,
            use_verbal_modality=args.use_verbal_modality,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            emotion_model_path=args.emotion_model_path,
            temporal_augment=args.temporal_augment,
        )
        if code != 0:
            failed.append(fold)

    print("\n=== LOSO run finished ===")
    if failed:
        print(f"Failed folds: {failed}")
        raise SystemExit(1)
    print("All requested folds completed successfully.")


if __name__ == "__main__":
    main()
