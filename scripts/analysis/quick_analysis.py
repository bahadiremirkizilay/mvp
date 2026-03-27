"""
Quick visualization to understand why predictions are so bad
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results_path = Path("results/batch_validation_results.csv")
df = pd.read_csv(results_path)

print("=" * 70)
print("PERFORMANCE ANALYSIS - 5 SUBJECTS")
print("=" * 70)

# Overall statistics
print(f"\n📊 Overall Metrics:")
print(f"   MAE:         {df['mae'].mean():.2f} ± {df['mae'].std():.2f} BPM")
print(f"   RMSE:        {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f} BPM")
print(f"   Correlation: {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}")
print(f"   Total samples: {df['n_samples'].sum()}")

# Per-subject analysis
print(f"\n📋 Per-Subject Breakdown:")
for _, row in df.iterrows():
    status = "✅" if row['mae'] < 10 else "⚠️" if row['mae'] < 15 else "❌"
    print(f"   {status} {row['subject_id']:10s}: MAE={row['mae']:6.2f}, Corr={row['correlation']:6.3f}, N={int(row['n_samples']):3d}")

# Prediction vs GT statistics
print(f"\n🎯 Mean BPM Statistics:")
print(f"   Predicted: {df['mean_pred'].mean():.1f} ± {df['mean_pred'].std():.1f} BPM")
print(f"   Ground Truth: {df['mean_gt'].mean():.1f} ± {df['mean_gt'].std():.1f} BPM")
print(f"   Bias: {(df['mean_pred'].mean() - df['mean_gt'].mean()):.1f} BPM")

# Identify issues
print(f"\n🔍 Potential Issues:")
bias = df['mean_pred'].mean() - df['mean_gt'].mean()
if abs(bias) > 5:
    print(f"   ⚠️ Systematic bias: {bias:.1f} BPM ({'under' if bias < 0 else 'over'}-estimating)")
else:
    print(f"   ✓ No systematic bias ({bias:.1f} BPM)")

if df['correlation'].mean() < 0.3:
    print(f"   ❌ Very poor correlation ({df['correlation'].mean():.3f}) - algorithm not tracking GT")
else:
    print(f"   ✓ Correlation: {df['correlation'].mean():.3f}")

variance_ratio = df['std_pred'].mean() / df['std_gt'].mean()
print(f"   Variance ratio (pred/GT): {variance_ratio:.2f}x")
if variance_ratio > 2:
    print(f"      ⚠️ Predictions too variable")
elif variance_ratio < 0.5:
    print(f"      ⚠️ Predictions too smooth (over-filtering?)")

# Recommendations
print(f"\n💡 Recommendations:")
print(f"   1. Visualize 1 subject's BPM over time (see prediction pattern)")
print(f"   2. Check if motion filtering is too aggressive")
print(f"   3. Test with less strict quality thresholds")
print(f"   4. Compare Level 1 (no filtering) vs Level 2")
print(f"   5. Verify POS algorithm implementation")

# Check which subject performed best
best_subject = df.loc[df['mae'].idxmin(), 'subject_id']
worst_subject = df.loc[df['mae'].idxmax(), 'subject_id']

print(f"\n🏆 Best subject: {best_subject} (MAE={df['mae'].min():.2f} BPM)")
print(f"   💡 Use this subject for debugging and visualization")
print(f"\n❌ Worst subject: {worst_subject} (MAE={df['mae'].max():.2f} BPM)")
print(f"   💡 Analyze what went wrong here")

print("\n" + "=" * 70)
print("Next step: Visualize BPM over time for subject5 (best performer)")
print("Command: python scripts/analysis/visualize_single_subject.py subject5")
print("=" * 70)
