"""
Build normalized deception manifests from Real-life Deception dataset folders.

Expected dataset structure:
    Real-life Deception Detection Dataset With Train Test/
        Train/
            trial_lie_001.mp4
            trial_truth_001.mp4
        Test/
            trial_lie_056.mp4
            trial_truth_055.mp4

Outputs:
    1) Full manifest with metadata for all parsable videos
    2) Filtered manifest that keeps videos above quality thresholds
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, List

import cv2


PAT = re.compile(r"trial_(lie|truth)_(\d+)\.mp4$", re.IGNORECASE)


def parse_label(name: str):
    m = PAT.search(name)
    if not m:
        return None
    label = m.group(1).lower()
    trial_id = int(m.group(2))
    label_id = 1 if label == "lie" else 0
    return label, label_id, trial_id


def get_video_meta(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    cap.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return {
        "opened": opened,
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration,
    }


def collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split_dir in ["Train", "Test"]:
        d = root / split_dir
        if not d.exists():
            continue
        split = split_dir.lower()
        for fp in sorted(d.glob("*.mp4")):
            parsed = parse_label(fp.name)
            if parsed is None:
                continue
            label, label_id, trial_id = parsed
            meta = get_video_meta(fp)
            rows.append({
                "split": split,
                "video_path": str(fp.as_posix()),
                "label": label,
                "label_id": label_id,
                "trial_id": trial_id,
                "dataset": "real_life_deception",
                "fps": round(meta["fps"], 4),
                "frame_count": meta["frame_count"],
                "duration_sec": round(meta["duration_sec"], 4),
                "readable": int(bool(meta["opened"]) and meta["frame_count"] > 0 and meta["fps"] > 0),
            })
    return rows


def filter_rows(
    rows: List[Dict[str, object]],
    min_duration_sec: float,
    min_fps: float,
) -> List[Dict[str, object]]:
    keep = []
    for r in rows:
        if int(r["readable"]) != 1:
            continue
        if float(r["duration_sec"]) < min_duration_sec:
            continue
        if float(r["fps"]) < min_fps:
            continue
        keep.append(r)
    return keep


def main():
    parser = argparse.ArgumentParser(description="Build deception manifest CSV")
    parser.add_argument(
        "--root",
        type=str,
        default="Real-life Deception Detection Dataset With Train Test",
        help="Root folder of real-life deception dataset",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/self_collected/deception_manifest.csv",
        help="Output FULL manifest CSV path",
    )
    parser.add_argument(
        "--out_filtered",
        type=str,
        default="data/self_collected/deception_manifest_filtered.csv",
        help="Output FILTERED manifest CSV path",
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=8.0,
        help="Minimum video duration for filtered manifest",
    )
    parser.add_argument(
        "--min_fps",
        type=float,
        default=20.0,
        help="Minimum FPS for filtered manifest",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    rows = collect_rows(root)
    if not rows:
        raise RuntimeError("No valid .mp4 files parsed. Check folder and naming pattern.")

    out = Path(args.out)
    out_filtered = Path(args.out_filtered)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_filtered.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "video_path",
        "label",
        "label_id",
        "trial_id",
        "dataset",
        "fps",
        "frame_count",
        "duration_sec",
        "readable",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    filtered = filter_rows(rows, min_duration_sec=args.min_duration_sec, min_fps=args.min_fps)
    with out_filtered.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(filtered)

    n_lie = sum(1 for r in rows if r["label"] == "lie")
    n_truth = sum(1 for r in rows if r["label"] == "truth")
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")

    n_bad = sum(1 for r in rows if int(r["readable"]) != 1)
    n_short = sum(1 for r in rows if float(r["duration_sec"]) < args.min_duration_sec)
    n_low_fps = sum(1 for r in rows if float(r["fps"]) < args.min_fps)

    f_lie = sum(1 for r in filtered if r["label"] == "lie")
    f_truth = sum(1 for r in filtered if r["label"] == "truth")
    f_train = sum(1 for r in filtered if r["split"] == "train")
    f_test = sum(1 for r in filtered if r["split"] == "test")

    print("=" * 80)
    print("DECEPTION MANIFEST READY")
    print("=" * 80)
    print(f"Full rows: {len(rows)}")
    print(f"Full split: Train={n_train} | Test={n_test}")
    print(f"Full labels: Lie={n_lie} | Truth={n_truth}")
    print(f"Quality stats: unreadable={n_bad}, short(<{args.min_duration_sec}s)={n_short}, low_fps(<{args.min_fps})={n_low_fps}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Filtered split: Train={f_train} | Test={f_test}")
    print(f"Filtered labels: Lie={f_lie} | Truth={f_truth}")
    print(f"Full output: {out}")
    print(f"Filtered output: {out_filtered}")


if __name__ == "__main__":
    main()
