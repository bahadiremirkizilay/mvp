# Validation Results Analysis & Improvement Plan

## Current Results (POOR)
- **MAE**: 15.95 BPM (Target: <5 BPM)
- **RMSE**: 18.07 BPM (Target: <7 BPM)
- **Correlation**: -0.65 (Target: >0.85)
- **Mean Error**: -13.6 BPM (underprediction)

## Root Cause Analysis

### 1. **CRITICAL: Negative Correlation (-0.65)**
This suggests timestamp/alignment issue, NOT algorithm failure.

**Potential causes:**
- Ground truth timestamps might be frame indices, not seconds
- FPS mismatch (video: 29.26 FPS, config: 30 FPS)
- Interpolation direction inverted
- Phase shift in signal processing

**Fix Priority: URGENT**

### 2. **Systematic Underprediction (-13.6 BPM)**
System consistently measures 13-14 BPM lower than ground truth.

**Potential causes:**
- Peak detection threshold too strict
- Bandpass filter attenuating signal
- FFT frequency resolution issue
- Min peak distance too large

**Fix Priority: HIGH**

### 3. **High Variance (std: 8.31 vs 4.68)**
Predictions are noisy.

**Potential causes:**
- Motion confidence threshold too low (0.30)
- No signal quality filtering
- Short temporal normalization window (0.5s)
- ROI instability

**Fix Priority: MEDIUM**

---

## Improvement Strategy

### Phase 1: Fix Timestamp Alignment (CRITICAL)
```python
# Check if ground truth uses frame indices instead of seconds
# Current: assumes timestamps are in seconds
# Fix: verify ground truth format and adjust
```

### Phase 2: Optimize Peak Detection
```yaml
hrv:
  min_peak_prominence: 0.15  # Lower from 0.2 (detect more peaks)
  min_peak_distance_sec: 0.40  # Keep at 0.45 or lower to 0.40
```

### Phase 3: Increase Signal Quality Threshold
```python
# Current: motion_conf >= 0.30
# Suggested: motion_conf >= 0.50
_MOTION_CONF_THRESHOLD = 0.50
```

### Phase 4: Optimize Signal Processing
```yaml
rppg:
  ma_window_sec: 0.5        # Increase from 0.3 (smoother)
  temporal_norm_window: 1.0  # Increase from 0.5s
```

### Phase 5: Bandpass Filter Tuning
```yaml
rppg:
  bandpass_low: 0.75   # Tighter range
  bandpass_high: 2.5   # Allow slightly higher
  bandpass_order: 6    # Sharper cutoff (from 4)
```

---

## Implementation Priority

1. ✅ **FIRST**: Debug timestamp alignment
   - Print ground truth timestamps vs predicted timestamps
   - Verify FPS calculation
   - Check interpolation logic

2. ✅ **SECOND**: Lower peak detection threshold
   - Test with min_prominence: 0.15, 0.12, 0.10
   - Test with min_peak_distance: 0.40, 0.35

3. ✅ **THIRD**: Increase motion confidence threshold
   - Test with: 0.40, 0.50, 0.60
   - Compare prediction stability

4. ✅ **FOURTH**: Optimize signal processing windows
   - Test different MA window sizes
   - Test temporal normalization windows

5. ✅ **FIFTH**: Add signal quality filtering
   - Only accept predictions with SQI > 0.4
   - Reject low-confidence estimates

---

## Expected Improvements

### After Phase 1 (Timestamp Fix):
- Correlation: -0.65 → **+0.50 to +0.70**
- This alone will have the biggest impact

### After Phase 2 (Peak Detection):
- MAE: 15.95 → **8-12 BPM**
- Mean error: -13.6 → **-5 to -8 BPM**

### After Phases 3-5 (Signal Quality):
- MAE: 8-12 → **<5 BPM**
- RMSE: 18.07 → **<7 BPM**
- Correlation: +0.70 → **>0.85**
- Std: 8.31 → **<5.0**

---

## Testing Protocol

For each change:
1. Run validation: `python validate.py`
2. Record MAE, RMSE, Correlation
3. Create comparison table
4. Keep best configuration

---

## Notes

- UBFC-RPPG is a challenging dataset (varying lighting, motion)
- State-of-art methods achieve MAE ~5-8 BPM on UBFC
- Our target: MAE <5 BPM, Correlation >0.85
