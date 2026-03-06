"""
Multimodal Feature Builder  [Phase 4 — Not yet implemented]
============================================================
Will aggregate per-frame features from all modalities into a unified
feature vector for the downstream fusion model.

Planned feature groups:
    rPPG / physiological:
        hr_bpm, sdnn, rmssd, lf_hf_ratio, psd_peak_freq

    Emotion:
        emotion_probs (8-dim softmax), valence, arousal

    Behavioral:
        blink_rate, blink_duration_mean, perclos,
        gaze_x, gaze_y, gaze_dispersion,
        head_pitch, head_yaw, head_roll

    Temporal context:
        sliding statistics (mean, std, delta) over configurable windows
"""

# TODO (Phase 4): Implement FeatureBuilder


class FeatureBuilder:
    """Placeholder — implement in Phase 4."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FeatureBuilder is scheduled for Phase 4."
        )
