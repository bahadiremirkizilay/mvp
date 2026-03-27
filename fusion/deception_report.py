#!/usr/bin/env python3
"""Build pooled deception LOSO reports with calibration and threshold analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def load_fold_metrics(run_dir: Path) -> List[dict]:
    direct = run_dir / "metrics.json"
    if direct.exists():
        paths = [direct]
    else:
        paths = sorted(run_dir.glob("fold_*/metrics.json"))
    rows: List[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def pooled_from_confusion(rows: List[dict]) -> Dict[str, float]:
    tp = tn = fp = fn = 0.0
    for r in rows:
        tm = r.get("test_metrics", {})
        tp += float(tm.get("tp", 0.0))
        tn += float(tm.get("tn", 0.0))
        fp += float(tm.get("fp", 0.0))
        fn += float(tm.get("fn", 0.0))

    n = tp + tn + fp + fn
    acc = (tp + tn) / max(1.0, n)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    tnr = tn / max(1.0, tn + fp)
    bal_acc = 0.5 * (recall + tnr)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def collect_predictions(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_prob: List[float] = []

    for r in rows:
        pred = r.get("test_predictions", {})
        yt = pred.get("y_true", [])
        yp = pred.get("y_prob", [])
        if len(yt) != len(yp):
            continue
        y_true.extend([int(v) for v in yt])
        y_prob.extend([float(v) for v in yp])

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float64)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    if y_true.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / y_true.size) * abs(acc - conf)
    return float(ece)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    if y_true.size == 0:
        return {
            "best_f1": None,
            "best_balanced_accuracy": None,
            "grid": [],
        }

    thresholds = np.linspace(0.01, 0.99, 99)
    grid = []
    best_f1 = (-1.0, 0.5)
    best_bacc = (-1.0, 0.5)

    for th in thresholds:
        y_hat = (y_prob >= th).astype(np.int64)
        tp = float(np.sum((y_true == 1) & (y_hat == 1)))
        tn = float(np.sum((y_true == 0) & (y_hat == 0)))
        fp = float(np.sum((y_true == 0) & (y_hat == 1)))
        fn = float(np.sum((y_true == 1) & (y_hat == 0)))

        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        tnr = tn / max(1.0, tn + fp)
        bacc = 0.5 * (recall + tnr)
        acc = (tp + tn) / max(1.0, tp + tn + fp + fn)

        row = {
            "threshold": round(float(th), 2),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(bacc),
        }
        grid.append(row)

        if f1 > best_f1[0]:
            best_f1 = (f1, th)
        if bacc > best_bacc[0]:
            best_bacc = (bacc, th)

    return {
        "best_f1": {
            "threshold": float(best_f1[1]),
            "f1": float(best_f1[0]),
        },
        "best_balanced_accuracy": {
            "threshold": float(best_bacc[1]),
            "balanced_accuracy": float(best_bacc[0]),
        },
        "grid": grid,
    }


def summarize_run(run_dir: Path) -> Dict[str, object]:
    rows = load_fold_metrics(run_dir)
    best_vals = [float(r.get("best_val_loss", np.nan)) for r in rows]
    best_vals = [v for v in best_vals if not np.isnan(v)]

    out: Dict[str, object] = {
        "run_dir": str(run_dir),
        "fold_count": len(rows),
        "feature_mode": rows[0].get("feature_mode") if rows else None,
        "best_val_loss_mean": float(np.mean(best_vals)) if best_vals else None,
        "best_val_loss_std": float(np.std(best_vals)) if best_vals else None,
        "pooled_confusion_metrics": pooled_from_confusion(rows),
    }

    y_true, y_prob = collect_predictions(rows)
    out["prediction_count"] = int(y_true.size)

    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        out["calibration"] = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "ece_15_bins": float(expected_calibration_error(y_true, y_prob, n_bins=15)),
            "brier": float(brier_score_loss(y_true, y_prob)),
        }
        out["threshold_analysis"] = threshold_sweep(y_true, y_prob)
    else:
        out["calibration"] = {
            "roc_auc": None,
            "pr_auc": None,
            "ece_15_bins": None,
            "brier": None,
        }
        out["threshold_analysis"] = {
            "best_f1": None,
            "best_balanced_accuracy": None,
            "grid": [],
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pooled deception LOSO report")
    parser.add_argument("--smoke_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="checkpoints/deception_report_pooled.json")
    args = parser.parse_args()

    smoke = summarize_run(Path(args.smoke_dir))
    payload: Dict[str, object] = {"smoke": smoke}

    if args.real_dir:
        real = summarize_run(Path(args.real_dir))
        payload["real"] = real

        s = smoke.get("pooled_confusion_metrics", {})
        r = real.get("pooled_confusion_metrics", {})
        payload["comparison"] = {
            "delta_accuracy": float(r.get("accuracy", 0.0) - s.get("accuracy", 0.0)),
            "delta_f1": float(r.get("f1", 0.0) - s.get("f1", 0.0)),
            "delta_balanced_accuracy": float(r.get("balanced_accuracy", 0.0) - s.get("balanced_accuracy", 0.0)),
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Pooled Deception Report ===")
    print(f"Saved: {out_path}")
    print(f"Smoke folds: {smoke['fold_count']} | predictions: {smoke['prediction_count']}")
    if args.real_dir:
        print(f"Real folds: {payload['real']['fold_count']} | predictions: {payload['real']['prediction_count']}")


if __name__ == "__main__":
    main()
