"""Live progress monitor for LOSO run. Run this separately while the main run is going."""
import json, time, sys
from pathlib import Path

output_root = Path("checkpoints/deception_reallife2016_loso_v3_full")

def summarize():
    folds = sorted(output_root.glob("fold_*"))
    done = []
    for f in folds:
        m = f / "metrics.json"
        if m.exists():
            try:
                data = json.loads(m.read_text())
                tm = data.get("test_metrics", {})
                preds = data.get("test_predictions", {})
                prob = preds.get("y_prob", [0.5])[0]
                done.append({
                    "fold": data.get("fold"),
                    "acc": tm.get("accuracy", 0),
                    "f1": tm.get("f1", 0),
                    "tp": tm.get("tp", 0),
                    "tn": tm.get("tn", 0),
                    "fp": tm.get("fp", 0),
                    "fn": tm.get("fn", 0),
                    "prob": prob,
                    "true": preds.get("y_true", [0])[0],
                    "pred": preds.get("y_pred", [0])[0],
                })
            except Exception:
                pass

    if not done:
        print("No completed folds yet.")
        return

    n = len(done)
    total_folds = len(list(output_root.glob("fold_*")))
    
    tp = sum(d["tp"] for d in done)
    tn = sum(d["tn"] for d in done)
    fp = sum(d["fp"] for d in done)
    fn = sum(d["fn"] for d in done)
    n_pos = tp + fn
    n_neg = tn + fp
    
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    bal_acc = 0.5 * (tp / max(1, n_pos) + tn / max(1, n_neg))
    
    # AUC estimate (needs probs)
    try:
        from sklearn.metrics import roc_auc_score
        y_true = [d["true"] for d in done]
        y_prob = [d["prob"] for d in done]
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    print(f"\n{'='*60}")
    print(f"LOSO Progress: {n}/121 folds done ({total_folds} dirs)")
    print(f"{'='*60}")
    print(f"  Pooled Accuracy:          {acc:.4f}")
    print(f"  Pooled F1:                {f1:.4f}")
    print(f"  Pooled Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Pooled AUC (sklearn):     {auc:.4f}")
    print(f"  TP={int(tp)}  TN={int(tn)}  FP={int(fp)}  FN={int(fn)}")
    print(f"  (pos_test={int(n_pos)}, neg_test={int(n_neg)})")
    print()
    
    # Show last 5 folds
    print("Last 5 folds:")
    for d in done[-5:]:
        marker = "✓" if d["pred"] == d["true"] else "✗"
        print(f"  fold_{d['fold']:03d}: true={d['true']} pred={d['pred']} prob={d['prob']:.3f} {marker}")

summarize()
