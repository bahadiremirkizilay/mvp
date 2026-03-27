# Data Status Report (March 17, 2026)

## ✅ READY FOR TRAINING

### 1. **Real-life Deception Detection 2016 (Complete Dataset)** ⭐ NEW
- **Location**: `data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/`
- **Videos**: 121 total (61 deceptive, 60 truthful - **PERFECT BALANCE**)
- **Subfolders**: 
  - `Clips/Deceptive/` (61 videos)
  - `Clips/Truthful/` (60 videos)
  - `Annotation/All_Gestures_Deceptive and Truthful.csv` (41 columns: ID + 40 gesture features)
  - `Transcription/` (text transcripts for each video)
- **Video Quality**: Avg 28.0 sec, range 27.7-28.3 sec
- **Subjects**: 56 speakers (21 female, 35 male, ages 16-60)
- **Gesture Annotations**: Binary features for smile, laugh, frown, eye movement, eyebrow movement, head movement, etc.
- **Status**: ✅ Ready for fusion model training + gesture feature extraction
- **Advantage**: Organized structure + gesture ground truth + transcriptions
- **Next Step**: Build manifest with gesture features → train with behavioral module

### 1b. **Real-life Deception Detection (Original Trial Videos)**
- **Location**: `data/Real-life Deception Detection Dataset With Train Test/`
- **Videos**: 99 (after quality filtering: >=8s, >=20fps)
- **Labels**: 
  - Lies: 55
  - Truths: 44
- **Split**: Train 87 / Test 12
- **Status**: ✅ Also ready (but 2016 version above is better organized)
- **Note**: Duplicate/Alternative source of same dataset

---

### 2. **Bag of Lies (boxoflies)** ⭐ NEW
- **Location**: `data/boxoflies/`
- **Content**: 25 videos + 25 paired audio WAV files
- **Subjects**: 5 (Adri, Dario, Maria, Miguel, Tamai)
- **Labels** (from `lie_detection_wav.txt`):
  - Truths: 8
  - Lies: 17
- **Audio Quality**: Has dedicated WAV files with gender metadata
- **Status**: ✅ Ready for audio-specific fusion training
- **Advantage**: First dataset with dedicated audio + video pairing
- **Next Step**: Build manifest for boxoflies → train audio+video branch

---

## ⏳ NOT READY (BLOCKING FACTORS)

### 3. **Self-Collected Deception Data**
- **Location**: `data/self_collected/`
- **Current Status**: 0 subjects, 0 segments
- **Required**: 12+ subjects, 120+ truth clips, 120+ deception clips
- **Blocker**: Essential for true subject-independent LOSO validation
- **Template Available**: `data/self_collected/labels_template.json`

---

## ℹ️ AUXILIARY DATA (Not blocking, optional)

### 4. **Politifact Credentials Data**
- **Location**: `data/archive (5)/`
- **Files**: 
  - politifact.csv (raw)
  - politifact_clean.csv (deduplicated)
  - politifact_clean_binarized.csv (2-class labels)
- **Purpose**: Text credibility labels (separate NLP branch, not video labels)
- **Status**: Can integrate later as auxiliary late-fusion signal

---

## 📊 SUMMARY TABLE

| Dataset | Videos | Subjects | Truth | Lies | Audio | Gesture Annot | Status |
|---------|--------|----------|-------|------|-------|---------------|----- |
| Real-life 2016 | 121 | 56 | 60 | 61 | ❌ | ✅ (40 features) | ✅ Ready |
| Real-life Trial | 99 | ? | 44 | 55 | ❌ | ❌ | ✅ Ready |
| Bag of Lies | 25 | 5 | 8 | 17 | ✅ | ❌ | ✅ Ready |
| Self-collected | 0 | 0 | 0 | 0 | - | - | ❌ Blocking |
| Politifact | - | - | - | - | - | - | ℹ️ Auxiliary |

---

## 🎯 IMMEDIATE ACTIONS

### Priority 1 (Do NOW - Best Option)
**Run fusion training on Real-life 2016 dataset ALONE**
- 121 perfectly balanced videos (60 truth, 61 lie)
- 40 gesture features available from annotation CSV
- Can extract behavioral features (head pose, blink, gaze) via fusion pipeline
- Transcriptions available for potential NLP augmentation
- **NO quality filtering needed** - properly curated dataset

Steps:
1. Build manifest: `data/RealLifeDeceptionDetection.2016/.../deception_manifest.csv` with:
   - video_id, path, label (deceptive/truthful)
   - video_duration, fps
   - gesture_features (from annotation CSV dict)
   
2. Run LOSO training: `python fusion/train_fusion.py --dataset reallife_2016 --num_epochs 50`

### Priority 1b (Alternative - Combine All Data)
**Run fusion on Real-life 2016 + Bag of Lies combined**
- 146 total videos
- Real-life (60 truth, 61 lie) + Bag of Lies (8 truth, 17 lie) = 68 truth, 78 lie (imbalanced, use weight)
- Enables audio + video + gesture testing together
- Better for multimodal fusion learning

### Priority 2 (Parallel)
1. **Build Bag of Lies manifest** → `data/boxoflies/deception_manifest.csv`
   - Include audio WAV paths
   - Subject IDs from filenames
   
2. Collect self-collected data (needed for subject-independent LOSO)
   - OR contact Real-life 2016 authors for subject/trial metadata → enable true subject stratification

### Priority 3 (Optional)
- Politifact NLP branch (text-only, auxiliary fusion)

---

## 🔧 RECOMMENDED MANIFEST BUILDS

### 1. Real-life 2016 Manifest (Best Option - Clean Data)
```bash
# Build from gesture annotation CSV
python scripts/setup/build_deception_manifest.py \
  --input_dir "data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips" \
  --annotation_csv "data/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv" \
  --output "data/RealLifeDeceptionDetection.2016/deception_manifest.csv" \
  --min_duration_sec 0 \
  --min_fps 0
# Note: No filtering needed - dataset is already clean
```

### 2. Bag of Lies Manifest (Audio-Rich)
```bash
python scripts/setup/build_deception_manifest.py \
  --input_dir "data/boxoflies" \
  --labels_file "data/boxoflies/lie_detection_wav.txt" \
  --output "data/boxoflies/deception_manifest.csv" \
  --min_duration_sec 3 \
  --min_fps 20
```

This will create a manifest with:
- All 25 videos + audio paths
- Subject IDs extracted from filenames
- Gender from labels file
- Quality flags (duration, fps check)
