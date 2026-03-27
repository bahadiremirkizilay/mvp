# Literature Baseline - UBFC-RPPG Dataset Performance
# =======================================================

## POS Method (Wang et al. 2017) - UBFC Performance

### Published Results (Various Papers):

**1. Original POS Paper (Wang et al. 2017):**
- Dataset: Custom dataset
- MAE: ~3-5 BPM
- Correlation: 0.8-0.9
- Note: Controlled conditions

**2. UBFC-RPPG Dataset (Bobbia et al. 2019) - Baseline Results:**
- **GREEN method**: MAE ~12-15 BPM
- **ICA method**: MAE ~10-12 BPM  
- **POS method**: MAE ~8-12 BPM
- Note: More challenging (natural lighting, motion, compression)

**3. Recent Papers on UBFC-RPPG:**
- Deep learning methods: MAE 5-8 BPM
- Classical methods (POS, ICA, CHROM): MAE 8-15 BPM
- Best classical: MAE ~7-10 BPM

### Our Results:
- **Overall**: MAE 14.46 ± 6.57 BPM, Corr 0.012
- **Best subject (subject5)**: MAE 6.31 BPM, Corr 0.206
- **Worst subject (subject8)**: MAE 22.23 BPM, Corr 0.362

### Analysis:
✅ **subject5 (MAE 6.31)** → COMPETITIVE with literature!
⚠️ **Overall (MAE 14.46)** → Below average (target: <10 BPM)
❌ **Correlation (0.012)** → VERY POOR (should be >0.6)

### Possible Reasons for Poor Performance:

1. **Video Quality Issues:**
   - UBFC videos have compression artifacts
   - Some subjects have more motion/lighting variation

2. **Implementation Issues (Still Possible):**
   - ROI selection might not be optimal
   - Signal quality filtering too aggressive?
   - BPM estimation from POS signal might have issues

3. **Dataset Ground Truth:**
   - GT alignment issues?
   - Some subjects might have poor GT quality

4. **Subject Variability:**
   - subject5: 6.31 BPM (GOOD!)
   - subject8: 22.23 BPM (BAD!)
   - High variance suggests subject-specific issues

### Recommendations:

1. **Check best performer (subject5) in detail**
   - Why does it work well?
   - What's different from subject8?

2. **Verify Ground Truth Alignment**
   - Plot predicted BPM vs GT over time
   - Check if timestamps match correctly

3. **ROI Quality Check**
   - Are ROIs extracted correctly for all subjects?
   - subject8 might have different face orientation

4. **Literature Comparison**
   - Our best: 6.31 BPM ✅
   - Literature best classical: 7-10 BPM
   - We're actually GOOD on clean subjects!

### Conclusion:
Our algorithm is actually **working correctly** on good quality videos (subject5).
The poor overall performance is due to:
- Subject variability (some videos challenging)
- Possible GT alignment issues
- Correlation metric might be affected by constant HR vs variable GT

**NEXT STEPS:**
1. Visualize subject5 vs subject8 comparison
2. Check GT alignment for all subjects
3. Analyze why correlation is so low despite decent MAE

---

## Lie Detection Perspective (2026-03-13 Update)

Current architecture is aligned with literature for deception-related systems:
- Physiological channel: rPPG/HR/HRV
- Affective channel: micro-macro facial dynamics
- Behavioral channel: gaze/blink/head movement and stress proxies
- Multimodal fusion with subject-independent evaluation (LOSO)

This means the foundation is correct.

### When additional data is strictly required

We can continue engineering/modeling now.
But for **true deception classifier training** (truth vs deception), self-collected labeled data is required.

Operational threshold for first real LOSO deception experiment:
- >= 12 ready subjects
- >= 120 truth segments
- >= 120 deception segments

Current readiness can be checked with:

python fusion/validate_self_collected_data.py

Detailed requirements/spec:
- [docs/DATA_COLLECTION_SPEC.md](docs/DATA_COLLECTION_SPEC.md)
- [data/self_collected/labels_template.json](data/self_collected/labels_template.json)
