"""
Analyze Batch Validation Results
=================================
Visualize and compare performance across multiple subjects.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results from correct location
RESULTS_PATH = PROJECT_ROOT / 'results' / 'batch_validation_results.csv'
df = pd.read_csv(RESULTS_PATH)

print("="*70)
print("BATCH VALIDATION ANALYSIS")
print("="*70)
print(f"Total subjects: {len(df)}")
print()

# Overall statistics
print("OVERALL METRICS (Mean ± Std)")
print("-" * 70)
print(f"MAE:         {df['mae'].mean():.2f} ± {df['mae'].std():.2f} BPM")
print(f"RMSE:        {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f} BPM")
print(f"Correlation: {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}")
print(f"Samples:     {df['n_samples'].mean():.0f} ± {df['n_samples'].std():.0f}")
print()

# Best and worst
print("BEST & WORST PERFORMERS")
print("-" * 70)
best_mae = df.loc[df['mae'].idxmin()]
print(f"Best MAE:         {best_mae['subject_id']} ({best_mae['mae']:.2f} BPM)")
best_corr = df.loc[df['correlation'].idxmax()]
print(f"Best Correlation: {best_corr['subject_id']} ({best_corr['correlation']:.3f})")
worst_mae = df.loc[df['mae'].idxmax()]
print(f"Worst MAE:        {worst_mae['subject_id']} ({worst_mae['mae']:.2f} BPM)")
worst_corr = df.loc[df['correlation'].idxmin()]
print(f"Worst Correlation: {worst_corr['subject_id']} ({worst_corr['correlation']:.3f})")
print()

# Bias analysis
print("BIAS ANALYSIS (Predicted - Ground Truth)")
print("-" * 70)
df['bias'] = df['mean_pred'] - df['mean_gt']
for _, row in df.iterrows():
    bias_str = f"{row['bias']:+.2f}"
    print(f"{row['subject_id']:<12} Pred: {row['mean_pred']:6.2f} | GT: {row['mean_gt']:6.2f} | Bias: {bias_str:>7} BPM")
print(f"\nOverall Bias: {df['bias'].mean():.2f} ± {df['bias'].std():.2f} BPM")
print()

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. MAE comparison
ax1 = fig.add_subplot(gs[0, 0])
colors = ['green' if x < 10 else 'orange' if x < 12 else 'red' for x in df['mae']]
ax1.bar(df['subject_id'], df['mae'], color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(df['mae'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["mae"].mean():.2f}')
ax1.set_ylabel('MAE (BPM)', fontsize=11, fontweight='bold')
ax1.set_title('Mean Absolute Error per Subject', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim(0, max(df['mae']) * 1.2)

# 2. RMSE comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(df['subject_id'], df['rmse'], color='steelblue', edgecolor='black', linewidth=1.5)
ax2.axhline(df['rmse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["rmse"].mean():.2f}')
ax2.set_ylabel('RMSE (BPM)', fontsize=11, fontweight='bold')
ax2.set_title('Root Mean Squared Error per Subject', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim(0, max(df['rmse']) * 1.2)

# 3. Correlation comparison
ax3 = fig.add_subplot(gs[0, 2])
colors = ['green' if x > 0.3 else 'orange' if x > 0 else 'red' for x in df['correlation']]
ax3.bar(df['subject_id'], df['correlation'], color=colors, edgecolor='black', linewidth=1.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.axhline(df['correlation'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["correlation"].mean():.3f}')
ax3.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax3.set_title('Correlation per Subject', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# 4. Mean predicted vs GT
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(df))
width = 0.35
ax4.bar(x - width/2, df['mean_pred'], width, label='Predicted', color='coral', edgecolor='black', linewidth=1.5)
ax4.bar(x + width/2, df['mean_gt'], width, label='Ground Truth', color='lightblue', edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Mean BPM', fontsize=11, fontweight='bold')
ax4.set_title('Mean BPM: Predicted vs Ground Truth', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(df['subject_id'])
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# 5. Bias per subject
ax5 = fig.add_subplot(gs[1, 1])
colors = ['red' if x < -5 else 'orange' if x < 0 else 'lightgreen' if x < 5 else 'green' for x in df['bias']]
ax5.bar(df['subject_id'], df['bias'], color=colors, edgecolor='black', linewidth=1.5)
ax5.axhline(0, color='black', linestyle='-', linewidth=2)
ax5.axhline(df['bias'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["bias"].mean():.2f}')
ax5.set_ylabel('Bias (BPM)', fontsize=11, fontweight='bold')
ax5.set_title('Prediction Bias (Pred - GT)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# 6. Correlation vs MAE
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(df['correlation'], df['mae'], s=150, c=df['mae'], cmap='RdYlGn_r', 
                      edgecolors='black', linewidth=2, alpha=0.7)
for _, row in df.iterrows():
    ax6.annotate(row['subject_id'], (row['correlation'], row['mae']), 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
ax6.set_xlabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax6.set_ylabel('MAE (BPM)', fontsize=11, fontweight='bold')
ax6.set_title('MAE vs Correlation', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='MAE')

# 7. Number of samples
ax7 = fig.add_subplot(gs[2, 0])
ax7.bar(df['subject_id'], df['n_samples'], color='mediumpurple', edgecolor='black', linewidth=1.5)
ax7.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax7.set_title('Samples per Subject', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3, axis='y')

# 8. Standard deviation comparison
ax8 = fig.add_subplot(gs[2, 1])
x = np.arange(len(df))
width = 0.35
ax8.bar(x - width/2, df['std_pred'], width, label='Predicted', color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax8.bar(x + width/2, df['std_gt'], width, label='Ground Truth', color='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax8.set_ylabel('Std Dev (BPM)', fontsize=11, fontweight='bold')
ax8.set_title('Standard Deviation: Predicted vs GT', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(df['subject_id'])
ax8.legend()
ax8.grid(alpha=0.3, axis='y')

# 9. Performance summary table
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('tight')
ax9.axis('off')

table_data = []
table_data.append(['Subject', 'MAE', 'Corr', 'Bias'])
for _, row in df.iterrows():
    table_data.append([
        row['subject_id'],
        f"{row['mae']:.2f}",
        f"{row['correlation']:.3f}",
        f"{row['bias']:+.2f}"
    ])

table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header formatting
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
for i in range(1, len(table_data)):
    mae = df.iloc[i-1]['mae']
    if mae < 10:
        color = '#C6EFCE'  # Green
    elif mae < 12:
        color = '#FFEB9C'  # Yellow
    else:
        color = '#FFC7CE'  # Red
    
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax9.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('UBFC-RPPG Batch Validation Results', fontsize=16, fontweight='bold', y=0.995)

# Save figure
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'batch_validation_analysis.png'
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"✓ Analysis plot saved to: {OUTPUT_PATH}")

plt.show()

# Performance classification
print("PERFORMANCE CLASSIFICATION")
print("-" * 70)
excellent = df[df['mae'] < 8]
good = df[(df['mae'] >= 8) & (df['mae'] < 10)]
fair = df[(df['mae'] >= 10) & (df['mae'] < 12)]
poor = df[df['mae'] >= 12]

print(f"Excellent (MAE < 8):    {len(excellent)} subjects {list(excellent['subject_id']) if len(excellent) > 0 else '[]'}")
print(f"Good (8 ≤ MAE < 10):    {len(good)} subjects {list(good['subject_id'])}")
print(f"Fair (10 ≤ MAE < 12):   {len(fair)} subjects {list(fair['subject_id'])}")
print(f"Poor (MAE ≥ 12):        {len(poor)} subjects {list(poor['subject_id']) if len(poor) > 0 else '[]'}")
print()

print("="*70)
print("KEY FINDINGS")
print("="*70)
print("1. System shows consistent UNDERESTIMATION (mean bias: {:.2f} BPM)".format(df['bias'].mean()))
print("2. MAE ranges from {:.2f} to {:.2f} BPM".format(df['mae'].min(), df['mae'].max()))
print("3. Correlation is HIGHLY VARIABLE: {:.3f} to {:.3f}".format(df['correlation'].min(), df['correlation'].max()))
print("4. Best performer: {} (MAE={:.2f}, Corr={:.3f})".format(best_mae['subject_id'], best_mae['mae'], best_corr['correlation']))
print("5. Subjects with negative correlation: {}".format(list(df[df['correlation'] < 0]['subject_id'])))
print()

print("RECOMMENDATIONS:")
print("-" * 70)
if df['bias'].mean() < -5:
    print("• Strong underestimation bias detected")
    print("  → Consider adjusting peak detection parameters")
    print("  → Check if bandpass filter is too narrow")
elif df['bias'].mean() > 5:
    print("• Strong overestimation bias detected")
    print("  → Peak detection may be too sensitive")

if df['correlation'].mean() < 0.3:
    print("• Low overall correlation")
    print("  → Motion artifacts may be significant (Dataset 2 is realistic with movement)")
    print("  → Consider stricter motion confidence threshold")
    print("  → ROI selection may need refinement")

if df['correlation'].std() > 0.3:
    print("• High variability in correlation across subjects")
    print("  → System performance is subject-dependent")
    print("  → Individual tuning or adaptive parameters may help")

print("="*70)
