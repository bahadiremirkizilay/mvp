# Lie Detection Roadmap (Literature-Aligned)

## 1) Are We on the Right Foundation?

Short answer: **yes**.

Current direction matches the dominant literature trend for deception-related affective systems:
- **Physiological channel**: HR/BPM, HRV, signal quality (rPPG)
- **Affective channel**: micro/macro facial expression analysis
- **Behavioral channel**: gaze, blink, head pose, stress proxies
- **Optional vocal channel**: prosody / speech rhythm / pitch stability / pause patterns
- **Multimodal fusion**: combine independent channels, evaluate subject-independent generalization

This is a correct foundation. The only major missing element for final "lie detection" is **deception-labeled ground truth data**.

## 2) Current Project Status (As of 2026-03-13)

Implemented and working:
- rPPG extraction and HRV/stress rule-based estimation in [main.py](main.py) and [behavioral/stress.py](behavioral/stress.py)
- SAMM micro-expression dataset with configurable augmentation in [emotion/samm_dataset.py](emotion/samm_dataset.py)
- Unified dataset scaffold in [emotion/unified_dataset.py](emotion/unified_dataset.py)
- Multimodal feature builder skeleton in [fusion/feature_builder.py](fusion/feature_builder.py)
- Baseline fusion model implemented in [fusion/fusion_model.py](fusion/fusion_model.py)
- Baseline fusion training loop implemented and smoke-tested in [fusion/train_fusion.py](fusion/train_fusion.py)

Not implemented yet (critical for end-goal):
- Real paired multimodal training dataset with deception labels
- LOSO evaluation script on real feature windows

## 3) Why Data Still Matters

You can continue now without blocking.

But for final deception classifier quality, you will need:
- **Task-level labels**: truthful vs deceptive (or deception probability)
- **Synchronized windows**: timestamps aligned across modalities
- **Subject diversity**: enough participants for LOSO robustness

Without deception labels, we can build a strong **lie-risk proxy**, not a clinically valid lie detector.

## 4) Updated Phased Plan

## Phase A - Stabilize Modality Outputs (Now)
Goal: make each stream reliable and calibrated before fusion.

Tasks:
- Lock SAMM augmentation defaults (already done)
- Validate rPPG signal quality gates and missing-data handling
- Export aligned per-window features from all modalities
- Keep audio optional in data schema so it can be fused later without changing label format

Deliverable:
- One merged feature table per session/window (CSV/Parquet)

Success criteria:
- >95% windows with valid feature rows
- No NaN leakage into training set

## Phase B - Build Fusion Baseline (Next)
Goal: train first multimodal model for stress/affect + lie-risk proxy.

Tasks:
- Implement baseline Fusion MLP/Transformer-lite in [fusion/fusion_model.py](fusion/fusion_model.py)
- Implement training/evaluation loop in [fusion/train_fusion.py](fusion/train_fusion.py)
- Add LOSO split protocol and weighted metrics (F1, AUC)

Deliverable:
- Reproducible training script + saved best checkpoint

Success criteria:
- Stable convergence in LOSO
- Meaningful uplift over best single modality

## Phase C - Lie-Risk Proxy Layer (Immediately After Fusion)
Goal: map multimodal evidence to interpretable risk score before full deception supervision.

Tasks:
- Build interpretable score from:
  - HR/BPM variability drift
  - HRV stress index
  - micro/macro intensity discordance
  - behavioral instability (blink/gaze/headpose)
  - vocal instability when audio is available
- Add uncertainty/confidence and low-quality rejection logic

Deliverable:
- `risk_score` in [0,1] + reason codes per window

Success criteria:
- Human-interpretable traces
- Reduced false alarms on low-signal windows

## Phase D - True Lie Detection Supervision
Goal: transition from proxy to real deception model.

Tasks:
- Collect/import deception-labeled sessions
- Train binary (truth/deception) or ordinal model
- Calibrate threshold by use-case (high precision vs high recall)

Deliverable:
- Deception model with calibrated operating point

Success criteria:
- Subject-independent metrics (LOSO) at target threshold
- Reliability analysis by domain conditions (lighting, motion, speaking)

## 5) Immediate Execution Queue (What We Do Next)

1. Implement baseline fusion model in [fusion/fusion_model.py](fusion/fusion_model.py)
2. Implement fusion training/eval in [fusion/train_fusion.py](fusion/train_fusion.py)
3. Add standardized feature export + alignment checks in [fusion/feature_builder.py](fusion/feature_builder.py)
4. Run first LOSO fusion experiment and publish metrics

## 6) Evaluation Protocol (Locked)

Primary protocol:
- **LOSO (Leave-One-Subject-Out)**

Primary metrics:
- Macro-F1
- Weighted-F1
- ROC-AUC (for binary/one-vs-rest)
- Calibration error (for risk outputs)

Quality controls:
- Signal-quality-aware masking
- Class imbalance handling (weights / focal options)

## 7) Risk Note

This system should be positioned as **decision support**, not a standalone legal/clinical lie detector.
Human-in-the-loop review is mandatory for deployment contexts.
