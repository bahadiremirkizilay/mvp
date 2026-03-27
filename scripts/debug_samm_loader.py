#!/usr/bin/env python3
"""Debug SAMM dataset loader to see why only 138/159 sequences load."""

import pandas as pd
from pathlib import Path

# Load annotations same way as dataset loader
annotation_path = Path("data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx")
df = pd.read_excel(annotation_path, engine='openpyxl', skiprows=12, header=0)

print("="*80)
print("SAMM ANNOTATION PARSING DEBUG")
print("="*80)

print(f"\nInitial shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:8]}")

# Check first few rows
print("\nFirst 3 rows:")
print(df.head(3))

# Remove completely empty rows
df_clean = df.dropna(how='all')
print(f"\nAfter dropna(how='all'): {df_clean.shape}")

# Check Subject column
subject_col = df_clean.columns[0]
print(f"\nSubject column name: '{subject_col}'")
print(f"Subject unique values: {df_clean[subject_col].nunique()}")
print(f"Subject value counts:")
print(df_clean[subject_col].value_counts().head(10))

# Check for NaN subjects
nan_subjects = df_clean[df_clean[subject_col].isna()]
print(f"\nRows with NaN subject: {len(nan_subjects)}")

# Remove NaN subjects
df_valid = df_clean[df_clean[subject_col].notna()]
print(f"After removing NaN subjects: {df_valid.shape}")

# Check filename column
filename_col = df_valid.columns[1] if len(df_valid.columns) > 1 else None
if filename_col:
    print(f"\nFilename column: '{filename_col}'")
    print(f"Filename unique: {df_valid[filename_col].nunique()}")
    
# Count file system sequences
root = Path("data/SAMM")
subjects = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
total_sequences = sum(len(list(s.iterdir())) for s in subjects if s.is_dir())

print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Annotations in Excel: {len(df_valid)}")
print(f"Sequences in filesystem: {total_sequences}")
print(f"Missing: {total_sequences - len(df_valid)}")

# Check train/val split
train_size = int(0.7 * len(df_valid))
val_size = int(0.15 * len(df_valid))
test_size = len(df_valid) - train_size - val_size

print(f"\nTrain/Val/Test split (70/15/15):")
print(f"  Train: {train_size}")
print(f"  Val: {val_size}")
print(f"  Test: {test_size}")
print(f"  Total used: {train_size + val_size}")
print(f"  (Test set not used in training)")
