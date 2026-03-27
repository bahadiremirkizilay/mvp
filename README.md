# rPPG-Based Mental Wellness System

Multi-modal emotion and stress detection system using remote photoplethysmography (rPPG), facial expressions, and behavioral cues.

## 📊 Current Performance (Level 2 Optimization)

**Validation on UBFC-RPPG Dataset 2:**
- **MAE:** 9.52 BPM
- **Correlation:** 0.129
- **Best Subject:** subject1 (Correlation 0.818)

## 🗂️ Project Structure

```
mvp/
├── scripts/
│   ├── validation/          # Validation scripts
│   │   ├── batch_validate.py   # Multi-subject batch validation
│   │   └── validate.py         # Single-video validation
│   └── analysis/            # Analysis scripts
│       └── analyze_batch_results.py  # Statistical analysis & visualization
├── results/                 # Validation results
│   ├── batch_validation_results.csv
│   ├── batch_validation_analysis.png
│   └── archived/           # Old results
├── docs/                   # Documentation
├── config/                 # Configuration files
│   └── config.yaml
├── data/                   # Datasets
│   ├── ubfc/              # UBFC-RPPG Dataset 2
│   ├── pure/
│   ├── affectnet/
│   └── self_collected/
├── models/                # Pre-trained models
├── rppg/                  # rPPG core modules
│   ├── pos_method.py
│   ├── roi_extractor.py
│   ├── signal_processing.py
│   └── hrv.py
├── emotion/               # Emotion recognition
├── fusion/                # Multi-modal fusion
├── behavioral/            # Behavioral analysis
├── utils/                 # Utilities
├── main.py               # Main application
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Batch Validation (5 subjects)
```bash
python scripts/validation/batch_validate.py --subjects subject1 subject3 subject4 subject5 subject8
```

### 2. Single Video Validation
```bash
python scripts/validation/validate.py --video data/ubfc/subject1/vid.avi --ground_truth data/ubfc/subject1/ground_truth.txt
```

### 3. Analyze Results
```bash
python scripts/analysis/analyze_batch_results.py
```

## 🔧 Configuration (Level 2 Optimization)

**Key Features:**
- **3 ROI Fusion:** Forehead, left cheek, right cheek (quality-weighted)
- **Motion Filtering:** Confidence threshold 0.55, stability check 0.15
- **Outlier Rejection:** Relaxed threshold 35 BPM for dynamic HR
- **Physiological Range:** 40-180 BPM
- **Temporal Smoothing:** Median filter (8-sample window)

## 📈 Optimization Levels Tested

| Level | Description | MAE | Correlation | Status |
|-------|-------------|-----|-------------|--------|
| Baseline | Simple median smoothing | 9.71 | 0.062 | ❌ |
| Level 1 | Aggressive parameter changes | 32.46 | NaN | ❌ FAILED |
| Level 1.5 | Outlier rejection fix | 9.92 | 0.105 | ✅ Better |
| **Level 2** | **Motion filtering** | **9.52** | **0.129** | ✅ **BEST** |
| Level 3 | Signal quality weighting | 9.92 | 0.079 | ❌ Worse |
| Level 5 | ICA/PCA decomposition | 10.88 | 0.024 | ❌ Unstable |

## 📚 Dataset

**UBFC-RPPG Dataset 2:**
- 42 subjects performing time-sensitive mathematical tasks
- Realistic scenario with stress and movement
- Ground truth from contact PPG sensor
- Currently tested: 5 subjects (subject1, 3, 4, 5, 8)

## 🛠️ Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:
- OpenCV
- MediaPipe (v0.10.9)
- NumPy, SciPy, Pandas
- PyTorch (emotion recognition)

## 📝 Notes

- **rPPG Method:** POS (Plane-Orthogonal-to-Skin)
- **Face Detection:** MediaPipe Face Mesh (468 landmarks)
- **Window Size:** 7 seconds
- **Bandpass Filter:** 0.8-2.2 Hz (48-132 BPM)
- **Sampling Rate:** ~30 FPS (video-dependent)

## 📧 Contact

For questions or issues, please open an issue on GitHub.
