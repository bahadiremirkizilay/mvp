"""Compare two LOSO runs side by side."""
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

def load_run(root):
    root = Path(root)
    results = []
    for f in sorted(root.glob("fold_*")):
        m = f / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        preds = data.get("test_predictions", {})
        tm = data.get("test_metrics", {})
        results.append({
            "fold": data.get("fold"),
            "true": preds.get("y_true", [0])[0],
            "pred": preds.get("y_pred", [0])[0],
            "prob": preds.get("y_prob", [0.5])[0],
            "tp": tm.get("tp", 0), "tn": tm.get("tn", 0),
            "fp": tm.get("fp", 0), "fn": tm.get("fn", 0),
        })
    return results

def summarize(name, results):
    tp = sum(d["tp"] for d in results)
    tn = sum(d["tn"] for d in results)
    fp = sum(d["fp"] for d in results)
    fn = sum(d["fn"] for d in results)
    n_pos = tp + fn
    n_neg = tn + fp
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-8, prec + rec)
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    bal  = 0.5 * (tp / max(1, n_pos) + tn / max(1, n_neg))
    y_true = [d["true"] for d in results]
    y_prob = [d["prob"] for d in results]
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    print(f"\n{'='*55}")
    print(f"  {name}  (n={len(results)})")
    print(f"{'='*55}")
    print(f"  Accuracy:          {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy: {bal:.4f}  ({bal*100:.1f}%)")
    print(f"  F1:                {f1:.4f}")
    print(f"  ROC-AUC:           {auc:.4f}")
    print(f"  Precision:         {prec:.4f} | Recall: {rec:.4f}")
    print(f"  TP={int(tp)}  TN={int(tn)}  FP={int(fp)}  FN={int(fn)}")
    return {"acc": acc, "bal": bal, "f1": f1, "auc": auc}

r_full    = load_run("checkpoints/deception_reallife2016_loso_v3_full")
r_gesture = load_run("checkpoints/deception_reallife2016_loso_v3_gestureonly")

m1 = summarize("ALL MODALITIES  (gesture + audio + verbal)", r_full)
m2 = summarize("GESTURE ONLY    (no audio, no verbal)", r_gesture)

print(f"\n{'='*55}")
print("  Delta (all - gesture_only)")
print(f"{'='*55}")
for k in ["acc", "bal", "f1", "auc"]:
    d = m1[k] - m2[k]
    sign = "+" if d >= 0 else ""
    print(f"  {k:6s}: {sign}{d:.4f}  ({'audio+verbal helped' if d > 0.005 else 'neutral/hurt' if d < -0.005 else '~equal'})")
