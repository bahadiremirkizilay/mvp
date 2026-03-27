# Scripts - Usage Guide

## 📁 Folder Structure

```
scripts/
├── validation/          # Validation scripts
│   ├── batch_validate.py   # Multi-subject batch validation
│   └── validate.py         # Single-video validation
└── analysis/            # Analysis & visualization
    └── analyze_batch_results.py
```

---

## 🔍 Validation Scripts

### `validation/batch_validate.py`

**Purpose:** Validate rPPG system on multiple subjects simultaneously

**Usage:**
```bash
# Validate 5 subjects
python scripts/validation/batch_validate.py --subjects subject1 subject3 subject4 subject5 subject8

# Validate all subjects in dataset
python scripts/validation/batch_validate.py
```

**Output:**
- `results/batch_validation_results.csv` - Per-subject metrics (MAE, RMSE, Correlation)
- Console output with detailed statistics

**Parameters:**
- `--subjects`: Space-separated list of subject IDs (optional)
- Dataset path: `data/ubfc/` (hardcoded, edit if needed)

---

### `validation/validate.py`

**Purpose:** Detailed validation on a single video with frame-by-frame output

**Usage:**
```bash
python scripts/validation/validate.py \
    --video data/ubfc/subject1/vid.avi \
    --ground_truth data/ubfc/subject1/ground_truth.txt
```

**Output:**
- `validation_data.csv` - Frame-by-frame predictions and ground truth
- `validation_results.txt` - Summary statistics
- Console output with detailed metrics

**Parameters:**
- `--video`: Path to video file (required)
- `--ground_truth`: Path to ground truth file (required)

---

## 📊 Analysis Scripts

### `analysis/analyze_batch_results.py`

**Purpose:** Statistical analysis and visualization of batch validation results

**Usage:**
```bash
python scripts/analysis/analyze_batch_results.py
```

**Input:**
- `results/batch_validation_results.csv`

**Output:**
- `results/batch_validation_analysis.png` - 9-panel visualization:
  - Overall metrics distribution
  - Per-subject MAE comparison
  - Correlation distribution
  - Bias analysis (Predicted vs Ground Truth)
  - Best/worst performers
  - Performance classification

**Features:**
- Automatic outlier detection
- Statistical summary (mean ± std)
- Performance classification:
  - Excellent: MAE < 8 BPM
  - Good: 8 ≤ MAE < 10 BPM
  - Fair: 10 ≤ MAE < 12 BPM
  - Poor: MAE ≥ 12 BPM

---

## 🎯 Level 2 Configuration (Active)

Scripts are configured with **Level 2 optimization** (best performance):

**Key Features:**
```python
# Motion filtering
_MOTION_CONF_THRESHOLD = 0.55      # Increased confidence threshold
_SIGNAL_STABILITY_THRESHOLD = 0.15  # Frame-to-frame stability check

# Outlier rejection (relaxed for dynamic HR)
outlier_threshold = 35.0 BPM

# Physiological range
valid_range = (40, 180) BPM

# ROI fusion
rois = ["forehead", "left_cheek", "right_cheek"]
fusion_method = "quality_weighted"  # Weight by signal quality

# Temporal smoothing
bpm_history = deque(maxlen=8)
smoothing = "median"
```

---

## 📈 Expected Performance (Level 2)

Based on 5-subject validation:

| Metric | Value |
|--------|-------|
| Overall MAE | 9.52 BPM |
| Overall RMSE | 11.06 BPM |
| Overall Correlation | 0.129 |
| Best Subject (subject1) | Corr: 0.818 |
| Challenging Subject (subject4) | MAE: 12.37 BPM |

---

## 🛠️ Customization

### Change Dataset Path

Edit in scripts:
```python
# In batch_validate.py
dataset_path = Path("data/ubfc")  # Change this
```

### Modify Optimization Level

To revert to baseline or test other levels, edit these sections in both `batch_validate.py` and `validate.py`:

```python
# Line ~110-120
_MOTION_CONF_THRESHOLD = 0.55  # Adjust motion threshold
_SIGNAL_STABILITY_THRESHOLD = 0.15  # Adjust stability check

# Line ~220-230
outlier_threshold = 35.0  # Adjust outlier rejection
```

### Add New ROI

Edit in `rppg/roi_extractor.py`:
```python
# Add new ROI landmarks
"nose_bridge": [6, 197, 195, 5, ...]  # MediaPipe landmark IDs
```

Then update ROI loop in validation scripts.

---

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'mediapipe'`
```bash
pip install mediapipe==0.10.9
```

**Issue:** `FileNotFoundError: config/config.yaml`
- Ensure you're running from project root: `mvp/`
- Use full paths: `python scripts/validation/batch_validate.py`

**Issue:** Low correlation / High MAE
- Check video quality (lighting, resolution)
- Verify face detection (print landmark confidence)
- Increase motion confidence threshold (0.55 → 0.60)

**Issue:** NaN correlation
- Usually caused by too strict outlier rejection
- Increase outlier threshold (35 → 40 BPM)
- Check if `bpm_history` is filling up

---

## 📝 Notes

- All scripts use **Level 2 optimization** (best validated configuration)
- Scripts automatically handle MediaPipe import fallback (python.solutions → solutions)
- Progress indicators show every 20% during processing
- Results are automatically timestamped in CSV format

---

## 🔄 Workflow Example

```bash
# 1. Run batch validation
python scripts/validation/batch_validate.py --subjects subject1 subject3 subject4 subject5 subject8

# 2. Analyze results
python scripts/analysis/analyze_batch_results.py

# 3. Check detailed output
cat results/batch_validation_results.csv
open results/batch_validation_analysis.png
```
