# Deception Detection Training Roadmap (Phase D - Implementation)
**Status**: March 17, 2026 | Ready to Start  
**Foundation**: ROADMAP_LIE_DETECTION.md Phase D  
**Data Sources**: Real-life 2016 (121), Bag of Lies (25), self-collected (future)

---

## Executive Summary

**Goal**: Build production-ready binary deception classifier using multimodal signals.

**Current State**:
- ✅ Fusion model architecture (Transformer + multi-task heads)
- ✅ Feature extraction pipeline (rPPG, emotion, behavioral, audio)
- ✅ Training loop with LOSO capability
- ✅ Real-life 2016 dataset (121 balanced videos + gesture annotations)
- ✅ Bag of Lies (25 videos + audio WAV pairs)
- ❌ Self-collected data (0 subjects, blocking final LOSO validation)

**Timeline**: 4 weeks to alpha (weeks 1-2 with Real-life, weeks 3-4 with audio).

---

## Phase D.1 - Data Pipeline & Manifest Building
**Duration**: ~2-3 days  
**Owner**: Data engineering  
**Blocking**: Yes (all downstream work waits here)

### D.1.1 Build Real-life 2016 Manifest
**File**: `scripts/setup/build_reallife2016_manifest.py` (new)

**Input**:
- `data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/[Deceptive|Truthful]/*.mp4`
- `data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv`
- `data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Transcription/[Deceptive|Truthful]/*.txt`

**Output**: `data/RealLifeDeceptionDetection.2016/deception_manifest.csv`

**Columns**:
```
video_id (e.g., trial_lie_001),
video_path (full path to .mp4),
label (0=truth, 1=lie),
label_source (ruling: guilty/not-guilty/exoneration),
duration_sec,
fps,
num_frames,
gesture_id (gesture_001, 002, ...),
smile, laugh, scowl, frown, head_move, eye_move, ... (40 binary features from annotation CSV),
has_transcription (bool),
transcription_path (if available),
quality_flags (readable, duration_ok, fps_ok)
```

**Script Logic**:
1. Parse Deceptive/ → label=1, Truthful/ → label=0
2. Extract duration/fps from each MP4 metadata
3. Cross-reference gesture annotation CSV by video ID
4. Attach gesture features (40 boolean cols from CSV)
5. Check for transcription files
6. Output single merged CSV (121 rows)

**QA**:
- ✅ All 121 videos matched to manifest rows
- ✅ No NaN in gesture features (0 or 1)
- ✅ Label distribution verified: 61 lie, 60 truth
- ✅ All gesture annotation IDs present

---

### D.1.2 Build Bag of Lies Manifest
**File**: `scripts/setup/build_boxoflies_manifest.py` (new)

**Input**:
- `data/boxoflies/*.mp4`
- `data/boxoflies/*_audio.wav`
- `data/boxoflies/lie_detection_wav.txt` (truth/lie labels)

**Output**: `data/boxoflies/deception_manifest.csv`

**Columns**:
```
video_id (e.g., Ronda_1_Adri),
round (Ronda_X),
subject_id (Adri, Dario, Maria, Miguel, Tamai),
gender (male/female from labels file),
video_path (full path to .mp4),
audio_path (full path to *_audio.wav),
label (0=truth, 1=lie from lie_detection_wav.txt),
duration_sec,
fps,
num_frames,
audio_duration_sec,
quality_flags (readable, video_readable, audio_readable, duration_ok, fps_ok)
```

**Script Logic**:
1. Parse lie_detection_wav.txt: match audio to label and gender
2. Find paired .mp4 and .wav files in folder
3. Extract video metadata (ffprobe)
4. Extract audio metadata (librosa / soundfile)
5. Output single merged CSV (25 rows)

**QA**:
- ✅ All 25 videos matched to .wav pairs
- ✅ All 25 audio files readable
- ✅ Label distribution verified: 17 lie, 8 truth
- ✅ Gender/subject metadata correct

---

### D.1.3 Create Combined Manifest (Optional - for later)
**File**: `scripts/setup/merge_manifests.py` (new, use later)

**Purpose**: Union Real-life 2016 + Bag of Lies for combined training.

**Input**:
- `data/RealLifeDeceptionDetection.2016/deception_manifest.csv`
- `data/boxoflies/deception_manifest.csv`

**Output**: `data/combined_deception_manifest.csv`

**Logic**:
1. Harmonize column names (gesture features only in Real-life, audio only in BoL)
2. Union rows, fill NaN with False/None where appropriate
3. Output 146 rows total

**QA**:
- ✅ 146 total rows (121 + 25)
- ✅ Label distribution: 68 truth (60+8), 78 lie (61+17) - imbalanced, use class_weight
- ✅ No row duplication

---

## Phase D.2 - Feature Extraction & Dataset Loading
**Duration**: ~3-4 days  
**Owner**: ML pipeline engineering  
**Depends on**: D.1.1, D.1.2

### D.2.1 Create DeceptionDataset Class
**File**: `fusion/deception_dataset.py` (new)

**Purpose**: PyTorch Dataset for loading video → features → labels pipeline.

**Class**: `DeceptionDataset(VideoToFeaturesDataset)`

**Interface**:
```python
class DeceptionDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, feature_cache_dir, modalities=['rppg', 'emotion', 'behavioral', 'audio']):
        # Load manifest
        # Configure which modalities to extract
        # Create cache keys per video
        
    def __getitem__(self, idx):
        # Load cached NPZ or extract on-the-fly
        # Return: {
        #   'video_id': str,
        #   'label': 0 or 1,
        #   'features': {
        #       'rppg': shape (T, rppg_dim),
        #       'emotion': shape (T, emotion_dim),
        #       'behavioral': shape (T, behavioral_dim),
        #       'audio': shape (T, audio_dim) or None
        #   },
        #   'signal_quality_mask': shape (T,)
        # }
        
    def __len__(self):
        # Return num videos in manifest
```

**Modality Outputs**:
- `rppg`: (T, 4) = [BPM, RMSSD, SDNN, LF/HF]
- `emotion`: (T, 10) = [8-class probs, valence, arousal]
- `behavioral`: (T, 7) = [gaze_x, gaze_y, blink_rate, head_roll, head_pitch, head_yaw, stress_proxy]
- `audio`: (T, 6) = [energy, ZCR, pitch, voiced_ratio, spectral_centroid, spectral_bandwidth] or None

**Feature Caching**:
- Check `feature_cache_dir/video_id.npz`
- If not found, extract from video on-the-fly
- Save extracted features to cache for next run (speedup)

**QA**:
- ✅ All 121 Real-life videos load without error
- ✅ All 25 Bag of Lies videos load without error (audio optional)
- ✅ Feature shapes consistent per modality
- ✅ No NaN in feature vectors (or masked properly)
- ✅ Label distribution preserved

---

### D.2.2 Implement LOSO Split (Without Subject ID)
**File**: `fusion/loso_sampler.py` (new/extend)

**Purpose**: Leave-One-{Trial|Video}-Out cross-validation (not subject-based yet).

**Note**: Real-life 2016 doesn't expose subject identity in file paths.  
**Workaround**: Use trial-based LOSO first (leave one video out per fold).
- This gives 121 folds for Real-life
- ~5-10 folds for Bag of Lies
- Not ideal, but validates overfitting

**Class**: `LOSOSampler(torch.utils.data.Sampler)`

```python
class LOSOSampler(Sampler):
    def __init__(self, manifest_csv, fold_idx, total_folds=None, modality='rppg'):
        # If total_folds=None, use leave-one-out (len(manifest) folds)
        # fold_idx: which fold to use as test set
        # Returns train/val/test split for this fold
        
    def get_fold_split(self):
        # Returns {'train': [...], 'val': [...], 'test': [test_video_id]}
```

**Implementation Detail**:
- 60% train, 20% val, 20% test (per fold)
- Stratified by label (equal lie/truth in each split)
- Deterministic splitting by fold_idx

**Future Enhancement** (after self-collected data):
- Accept optional `subject_id_mapping.csv`
- If present, stratify by subject instead
- This will give 12+ folds for true subject-independent LOSO

---

## Phase D.3 - Training Pipeline (Real-life 2016)
**Duration**: ~5-7 days  
**Owner**: ML training  
**Depends on**: D.2.1, D.2.2

### D.3.1 Update train_fusion.py for Deception Task
**File**: `fusion/train_fusion.py` (modify existing)

**Changes**:
1. Add `--dataset` arg: `{synthetic|reallife_2016|boxoflies|combined|custom}`
2. Add `--modalities` arg: list of modalities to use
3. Add `--loss_weights` arg: per-task weighting (lie_risk has higher weight now)
4. Add `--use_gesture_features` flag: incorporate 40 gesture annotations as auxiliary input
5. Add `--loso_fold` arg: which fold to run (default 0, range [0, num_videos))

**Loss Function Update**:
```python
# Original (lie-risk proxy):
loss = w_affect * affect_loss + w_stress * stress_loss + w_cognitive * cognitive_loss + w_engagement * engagement_loss + w_lie_risk * lie_risk_loss

# New (deception supervision):
deception_loss = F.binary_cross_entropy_with_logits(
    logits=fusion_model.deception_head(features),
    target=labels,
    pos_weight=torch.tensor(len_truth / len_lie)  # handle imbalance
)
loss = deception_loss + 0.1 * auxiliary_loss  # auxiliary = multi-task heads from Phase B
```

**Training Loop**:
1. Load manifest → DeceptionDataset
2. Create LOSOSampler for fold_idx
3. Split into train/val/test sets
4. Initialize FusionModel with deception_head
5. Train for N epochs:
   - Epoch: train on train_set, eval on val_set
   - Early stop if val_loss doesn't improve
   - Save best checkpoint
6. Evaluate on test_set:
   - Compute F1, AUC, precision, recall, confusion matrix
   - Per-fold report

**Hyperparameters** (to tune):
```
learning_rate: 1e-4
batch_size: 16
max_epochs: 100
warmup_epochs: 5
early_stop_patience: 10
weight_decay: 1e-5
gradient_clip: 1.0
```

**Output**:
- `checkpoints/deception_reallife2016_fold0/best_model.pth`
- `checkpoints/deception_reallife2016_fold0/metrics.json`
- `checkpoints/deception_reallife2016_fold0/confusion_matrix.png`
- `checkpoints/deception_reallife2016_fold0/roc_auc.png`

---

### D.3.2 Run One LOSO Fold (Smoke Test)
**Command**:
```bash
python fusion/train_fusion.py \
  --dataset reallife_2016 \
  --manifest data/RealLifeDeceptionDetection.2016/deception_manifest.csv \
  --modalities rppg emotion behavioral \
  --loso_fold 0 \
  --num_epochs 100 \
  --batch_size 16 \
  --output_dir checkpoints/deception_reallife2016_fold0 \
  --device cuda
```

**Expected Output** (fold 0):
- Train set: ~97 videos (mix of truth/lie)
- Val set: ~12 videos
- Test set: 1 video (leave-one-out)
- Expected F1 on test: 0.4-0.6 (single video, high variance)
- Real F1 only meaningful after aggregating 121 folds

**QA**:
- ✅ No crashes on first fold
- ✅ Loss converges
- ✅ Checkpoints save correctly
- ✅ Metrics computed per fold

---

### D.3.3 Run Full LOSO (All 121 Folds)
**Command**:
```bash
for fold in {0..120}; do
    python fusion/train_fusion.py \
      --dataset reallife_2016 \
      --manifest data/RealLifeDeceptionDetection.2016/deception_manifest.csv \
      --modalities rppg emotion behavioral \
      --loso_fold $fold \
      --num_epochs 100 \
      --batch_size 16 \
      --output_dir checkpoints/deception_reallife2016_fold_$fold \
      --device cuda
done
```

**Runtime**: ~48-72 hours (121 folds × 20 min per fold on GPU)

**Aggregated Metrics** (after all folds):
- Macro F1, weighted F1, AUC
- Per-label F1 (truth vs lie)
- Confusion matrix (121 test predictions)
- Calibration curve
- Per-gesture feature importance (via attention weights)

---

## Phase D.4 - Audio Modality Testing (Bag of Lies)
**Duration**: ~3-4 days  
**Owner**: ML audio module  
**Depends on**: D.3.1 (completed smoke test)

### D.4.1 Train on Bag of Lies Only
**Command**:
```bash
python fusion/train_fusion.py \
  --dataset boxoflies \
  --manifest data/boxoflies/deception_manifest.csv \
  --modalities rppg emotion behavioral audio \
  --num_epochs 100 \
  --batch_size 4 \
  --output_dir checkpoints/deception_boxoflies_audio \
  --device cuda
```

**Expected**:
- Train: 20 videos (5 truth, 15 lie)
- Val: 3 videos
- Test: 2 videos
- High variance (only 25 samples), treat as exploratory

**Focus**:
- Does audio feature extraction work?
- Does multimodal fusion (4 modalities) train without error?
- Initial impression: audio helps or hurts?

**Output**:
- `checkpoints/deception_boxoflies_audio/best_model.pth`
- `checkpoints/deception_boxoflies_audio/feature_importance.json` → which modality has highest attention weight?

---

### D.4.2 Train on Combined Dataset (Real-life + Bag of Lies)
**Command** (requires D.1.3 first):
```bash
python fusion/train_fusion.py \
  --dataset combined \
  --manifest data/combined_deception_manifest.csv \
  --modalities rppg emotion behavioral audio \
  --class_weight auto \
  --num_epochs 100 \
  --batch_size 16 \
  --output_dir checkpoints/deception_combined_audio \
  --device cuda
```

**Expected**:
- Train: 116 videos
- Val: 15 videos
- Test: 15 videos
- Better statistical power than Bag of Lies alone
- Can evaluate: does audio help Real-life classif?

**Output**:
- `checkpoints/deception_combined_audio/best_model.pth`
- Ablation report: {rppg_only, emotion_only, behavioral_only, audio_only, all_4}

---

## Phase D.5 - Self-Collected Integration (Future)
**Duration**: ~5-7 days (when data arrives)  
**Owner**: ML + data engineering  
**Depends on**: self-collected data collection complete (12+ subjects, 240+ segments)

### D.5.1 Build Self-Collected Manifest
**File**: `scripts/setup/build_selfcollected_manifest.py` (implement when data ready)

**Logic**:
- Parse `data/self_collected/subject_*/sessions/*/labels.json`
- Extract: subject_id, session_id, label (truth/deception), timestamps
- Map to video clips (`subject_*/sessions/*/raw_video.mp4`)
- Output: `data/self_collected/deception_manifest.csv`

---

### D.5.2 Run Subject-Stratified LOSO
**Purpose**: True subject-independent generalization test.

**Logic**:
- For each subject in {1..N}:
  - Test set: all samples from subject
  - Train set: all samples from other subjects
  - Evaluate: can model generalize to new subject?

**Expected**:
- Subject-stratified F1 lower than video-stratified (harder task)
- But more trustworthy for deployment (real generalization)

---

## Phase D.6 - Evaluation & Reporting
**Duration**: ~3-4 days (after training complete)  
**Owner**: ML evaluation  
**Depends on**: D.3.3 (full LOSO), D.4.2 (combined training)

### D.6.1 Build Comprehensive Metrics Dashboard
**Output**: `reports/deception_evaluation_report.md`

**Sections**:
1. **Dataset Summary**
   - Real-life 2016: 121 videos, 60 truth, 61 lie
   - Bag of Lies: 25 videos, 8 truth, 17 lie
   - Combined: 146 videos, 68 truth, 78 lie

2. **Model Architecture**
   - Fusion model: Transformer encoder + attention pooling + deception head
   - Hyperparameters used

3. **LOSO Results (Real-life 2016)**
   - Macro F1 (average across 121 folds)
   - Weighted F1
   - AUC
   - Per-label F1 (truth vs lie)
   - Confusion matrix

4. **Ablation Study** (which modality matters most?)
   - rPPG alone
   - Emotion alone
   - Behavioral alone
   - Audio (Bag of Lies)
   - All 4 combined

5. **Gesture Feature Importance**
   - Correlation between gesture annotations and predictions
   - Which gestures are strongest lie indicators?

6. **Per-Video Analysis**
   - Top 5 easiest predictions (high confidence)
   - Top 5 hardest predictions (low confidence)
   - Failure cases analysis

7. **Recommendations**
   - Model readiness for deployment
   - Confidence thresholds per use-case
   - Next improvements (self-collected LOSO, more data, etc.)

---

### D.6.2 Create Visualization Suite
**Output**: `reports/deception_evaluation_plots.html` (interactive)

**Plots**:
1. ROC-AUC per fold + aggregated
2. Confusion matrix (121 predictions)
3. Calibration curve
4. Feature importance heatmap (all 4 modalities)
5. Per-gesture feature importance
6. Loss curves over epochs
7. Video-level predictions scatter (truth vs lie)

---

## Phase D.7 - Model Deployment Readiness
**Duration**: ~2-3 days  
**Owner**: ML ops  
**Depends on**: D.6 (evaluation complete)

### D.7.1 Create Inference Script
**File**: `inference_deception.py` (new)

**Interface**:
```python
classifier = DeceptionClassifier(
    model_path='checkpoints/deception_reallife2016_best.pth',
    manifest_cache='data/RealLifeDeceptionDetection.2016/deception_manifest.csv'
)

prediction = classifier.predict_video(
    video_path='path/to/test_video.mp4',
    return_modality_scores=True,  # per-modality confidence
    return_gesture_analysis=True   # gesture-based explanation
)
# Output: {
#   'deception_probability': 0.75,
#   'confidence': 0.92,
#   'modality_scores': {'rppg': 0.68, 'emotion': 0.82, 'behavioral': 0.71},
#   'gesture_references': ['sudden_eye_movement', 'blink_spike'],
#   'explanation': str
# }
```

---

### D.7.2 Version & Document Model
**File**: `models/deception_classifier_v1.0.md`

**Contents**:
- Model ID, version, creation date
- Training data: which datasets, which folds
- Performance metrics (F1, AUC)
- Modalities used
- Known limitations
- Intended use & ethical guidelines
- Update history

---

## Phase D.8 - Self-Collected LOSO Integration (Final Validation)
**Duration**: ~4-5 days (after D.5.2)  
**Owner**: ML evaluation  
**Depends on**: self-collected data + D.5.2

### D.8.1 Re-run LOSO on Combined Dataset (with subject stratification)
**Command**:
```bash
python fusion/train_fusion.py \
  --dataset combined \
  --manifest data/combined_deception_manifest.csv \
  --subject_id_mapping data/combined_subject_mapping.csv \
  --loso_strategy subject_stratified \
  --num_epochs 100 \
  --output_dir checkpoints/deception_combined_subject_loso \
  --device cuda
```

**Expected**:
- Subject-stratified LOSO folds: #subjects (12+)
- F1 lower than video-stratified (hard task)
- More trustworthy for deployment

**Final Report**: `reports/final_deception_evaluation_subject_loso.md`

---

## Execution Timeline

```
Week 1:
  Mon-Tue   → D.1.1, D.1.2 (manifests built)
  Wed       → D.2.1, D.2.2 (dataset loader ready)
  Thu       → D.3.1, D.3.2 (one fold smoke test)
  Fri       → Debug, fix failures

Week 2:
  Mon-Wed   → D.3.3 (run all 121 folds, ~48-72 hrs)
  Thu-Fri   → D.6.1, D.6.2 (evaluation report + plots)

Week 3:
  Mon-Tue   → D.4.1, D.4.2 (audio modality testing)
  Wed-Thu   → D.7.1, D.7.2 (deployment readiness)
  Fri       → Refinement & documentation

Week 4 (Optional - if self-collected arrives):
  Mon-Tue   → D.5.1 (self-collected manifest)
  Wed-Thu   → D.5.2, D.8.1 (subject-stratified LOSO)
  Fri       → Final report & deployment decision
```

---

## Success Criteria (Completion Definition)

### Minimum (MVP):
- ✅ Real-life 2016 LOSO complete (121 folds)
- ✅ Macro F1 >= 0.55 on test (better than chance)
- ✅ Model checkpoints saved
- ✅ Evaluation report generated

### Desirable:
- ✅ Bag of Lies audio testing complete
- ✅ Combined dataset training done
- ✅ Ablation study showing modality contributions
- ✅ Gesture feature importance analyzed

### Optimal (Before self-collected):
- ✅ All D.1-D.7 phases complete
- ✅ F1 >= 0.65 on Real-life LOSO
- ✅ Model deployable (inference script ready)
- ✅ Documentation complete

---

## Potential Blockers & Mitigation

| Blocker | Risk | Mitigation |
|---------|------|-----------|
| Feature extraction crashes | High | Test on 5 sample videos first (D.2.1 QA) |
| LOSO training too slow | Medium | Use smaller batch_size, fp16 precision, distributed training |
| Audio features missing | Low | Fallback to video+rPPG+behavioral if audio fails |
| Self-collected delays | Medium | Proceed with video-stratified LOSO, re-run subject-stratified later |
| Overfitting on small Real-life | Medium | Use strong regularization, early stopping, data augmentation |
| Imbalanced labels (Bag of Lies) | Low | Use class_weight in loss |

---

## Next Action

**START HERE**: D.1.1 - Build Real-life 2016 manifest

```bash
# Step 1: Create script
python -c "
# Will implement build_reallife2016_manifest.py
"

# Step 2: Verify output
head -5 data/RealLifeDeceptionDetection.2016/deception_manifest.csv
wc -l data/RealLifeDeceptionDetection.2016/deception_manifest.csv  # Should be 122 (header + 121 rows)
```

Ready to start? 🚀
