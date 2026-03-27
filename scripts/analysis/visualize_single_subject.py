"""
Visualize BPM predictions vs ground truth for best performing subject
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def analyze_subject(subject_name="subject5"):
    """Analyze a single subject from results CSV"""
    
    results_path = Path("results/batch_validation_results.csv")
    df = pd.read_csv(results_path)
    
    subject_row = df[df['subject_id'] == subject_name]
    
    if len(subject_row) == 0:
        print(f"ERROR: Subject {subject_name} not found in results")
        return
    
    row = subject_row.iloc[0]
    
    print(f"\n{'='*60}")
    print(f"Detailed Analysis: {subject_name}")
    print(f"{'='*60}")
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE:         {row['mae']:.2f} BPM")
    print(f"   RMSE:        {row['rmse']:.2f} BPM")
    print(f"   Correlation: {row['correlation']:.3f} (p={row['p_value']:.4f})")
    print(f"   Samples:     {int(row['n_samples'])}")
    
    print(f"\n🎯 Mean BPM:")
    print(f"   Predicted:     {row['mean_pred']:.1f} BPM")
    print(f"   Ground Truth:  {row['mean_gt']:.1f} BPM")
    print(f"   Bias:          {row['mean_pred'] - row['mean_gt']:.1f} BPM")
    
    print(f"\n📈 Variability:")
    print(f"   Predicted Std: {row['std_pred']:.2f} BPM")
    print(f"   GT Std:        {row['std_gt']:.2f} BPM")
    print(f"   Ratio:         {row['std_pred'] / row['std_gt']:.2f}x")
    
    # Analysis
    print(f"\n🔍 Diagnosis:")
    
    bias = row['mean_pred'] - row['mean_gt']
    if abs(bias) > 5:
        print(f"   ⚠️ Large systematic bias: {bias:.1f} BPM")
    else:
        print(f"   ✅ Bias acceptable: {bias:.1f} BPM")
    
    if row['correlation'] < 0.3:
        print(f"   ❌ Poor correlation: {row['correlation']:.3f}")
    elif row['correlation'] < 0.6:
        print(f"   ⚠️ Moderate correlation: {row['correlation']:.3f}")
    else:
        print(f"   ✅ Good correlation: {row['correlation']:.3f}")
    
    var_ratio = row['std_pred'] / row['std_gt']
    if var_ratio > 2:
        print(f"   ⚠️ Predictions too variable ({var_ratio:.2f}x)")
    elif var_ratio < 0.5:
        print(f"   ⚠️ Predictions too smooth ({var_ratio:.2f}x) - over-filtering?")
    else:
        print(f"   ✅ Variance reasonable ({var_ratio:.2f}x)")
    
    print(f"\n💡 Recommendations:")
    if bias < -5:
        print(f"   - Systematic underestimation → Check bandpass filter cutoff")
        print(f"   - Try higher highcut frequency (currently 4 Hz = 240 BPM)")
    if row['correlation'] < 0.3:
        print(f"   - Algorithm not tracking GT → Check POS implementation")
        print(f"   - Try Level 0 or Level 1 (less filtering)")
    if var_ratio > 2:
        print(f"   - Too much variance → Increase motion threshold")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "subject5"
    analyze_subject(subject)

