# Required Datasets for This Project

This file answers: "If I can collect only a few datasets, which ones are mandatory?"

## Priority 1 (Mandatory)

1. Real-life deception video dataset (truth/deception labeled)
- Purpose: core lie/deception supervision for video-based model
- Minimum target: >= 120 truth + >= 120 deception segments
- Minimum subject target for robust validation: >= 12 subjects

2. Self-collected deception dataset with protocol labels
- Purpose: domain adaptation + subject diversity + real deployment conditions
- Must include: video.mp4 + labels.json (truth/deception timestamps)
- Optional but recommended: audio.wav, ecg_reference.csv

## Priority 2 (Strongly Recommended)

3. Bag of Lies (audio-rich deception)
- Purpose: strengthen audio/vocal deception branch
- Role: auxiliary modality training (not sole main dataset)
- Recommended storage: external disk + project link (junction)

4. SAMM / micro-expression dataset
- Purpose: improve subtle facial dynamics branch
- Role: pretraining/auxiliary for micro-expression sensitivity

## Priority 3 (Optional, Complementary)

5. Politifact CSV (text fact-check)
- Purpose: NLP truthfulness prior from statement text
- Important: does NOT replace visual/physio deception labels
- Usage: separate text branch, then late-fusion

6. rPPG public datasets (UBFC, PURE)
- Purpose: improve pulse/HRV robustness under varied conditions
- Role: rPPG component calibration and validation

## Practical Recommendation

If bandwidth/time is limited, do this first:
1. Real-life deception + self-collected labels
2. Add Bag of Lies for audio
3. Add SAMM for micro-expression refinement
4. Keep Politifact as optional text branch
