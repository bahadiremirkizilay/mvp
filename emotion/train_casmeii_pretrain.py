"""Train ResNet-18 on CASME II for facial emotion recognition.

The resulting model is used as a feature extractor in the deception detection
pipeline.  It outputs per-frame 7-class softmax probabilities which are far more
generalizable than pixel-level statistics.

Usage:
    python emotion/train_casmeii_pretrain.py
    python emotion/train_casmeii_pretrain.py --epochs 40 --lr 5e-4
    python emotion/train_casmeii_pretrain.py --output_dir checkpoints/emotion/casmeii_pretrain_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CASME_DIR = ROOT / "data" / "CASMEII"

# Unified label map for all CASME II folder-name variants
LABEL_MAP = {
    "angry": 0, "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3, "happiness": 3,
    "neutral": 4,
    "sad": 5, "sadness": 5,
    "surprise": 6,
}
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CASMEIIDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.items: list[tuple[str, int]] = []
        if not split_dir.exists():
            raise FileNotFoundError(f"CASME II split directory not found: {split_dir}")
        for label_folder in sorted(split_dir.iterdir()):
            if not label_folder.is_dir():
                continue
            label_name = label_folder.name.lower().strip()
            if label_name not in LABEL_MAP:
                continue
            label_idx = LABEL_MAP[label_name]
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in label_folder.glob(ext):
                    self.items.append((str(img_path), label_idx))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _class_weights(items: list, num_classes: int, device: str) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in items:
        counts[label] += 1
    w = 1.0 / (counts + 1e-6)
    w /= w.sum() / num_classes   # keep scale reasonable
    return torch.tensor(w, dtype=torch.float32).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train ResNet-18 on CASME II for deception pipeline emotion features")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--backbone",    default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--output_dir",  default="checkpoints/emotion/casmeii_pretrain_v2")
    parser.add_argument("--casme_dir",   default=str(CASME_DIR))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    casme_root = Path(args.casme_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CASMEIIDataset(casme_root / "train", transform=train_tfm)
    val_ds   = CASMEIIDataset(casme_root / "test",  transform=val_tfm)

    if len(train_ds) == 0:
        print(f"ERROR: No training images found in {casme_root}/train")
        print("Expected folder structure: data/CASMEII/train/<emotion>/*.jpg")
        raise SystemExit(1)

    print(f"Train: {len(train_ds)} images   Val: {len(val_ds)} images")

    # Per-class counts for info
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for _, lbl in train_ds.items:
        counts[lbl] += 1
    for name, cnt in zip(CLASS_NAMES, counts):
        print(f"  {name:10s}: {cnt}")

    class_w = _class_weights(train_ds.items, NUM_CLASSES, device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------------------------------------------- model
    if args.backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze only the very first layers — they detect edges/textures which transfer well.
    # We fine-tune from layer2 onward for domain adaptation.
    for name, param in model.named_parameters():
        if name.startswith("layer1") or name.startswith("conv1") or name.startswith("bn1"):
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.backbone}  total={total_params:,}  trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # --------------------------------------------------------------- training
    best_val_acc = 0.0
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        train_acc = train_correct / max(1, train_total)
        val_acc   = val_correct   / max(1, val_total)
        lr_cur    = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_acc:.3f}  val={val_acc:.3f} | lr={lr_cur:.6f}")
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch":            epoch,
                "backbone":         args.backbone,
                "num_classes":      NUM_CLASSES,
                "class_names":      CLASS_NAMES,
                "val_acc":          val_acc,
                "model_state_dict": model.state_dict(),
            }
            torch.save(ckpt, out_dir / "best_model.pth")
            print(f"  ✓ New best val_acc={val_acc:.3f} — saved")

    # ---------------------------------------------------------------- summary
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc, "backbone": args.backbone, "num_classes": NUM_CLASSES}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done.  Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
