"""
Blink Detection  [Phase 3 — Not yet implemented]
=================================================
Will estimate the Eye Aspect Ratio (EAR) from MediaPipe Face Mesh
landmarks to detect blink events and compute blink-rate features for
the multimodal fusion stage.

Planned metrics:
    • blink_rate_per_min   — blinks / minute (fatigue / cognitive load proxy)
    • blink_duration_ms    — mean / std of blink duration
    • perclos               — proportion of time eyes are ≥ 80% closed
                              (drowsiness indicator, PERCLOS standard)

Reference:
    Veltri, E., et al. (2022). Eye Aspect Ratio for Blink Detection.
    Sensors, 22(17), 6467.
"""

# TODO (Phase 3): Implement BlinkDetector


class BlinkDetector:
    """Placeholder — implement in Phase 3."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BlinkDetector is scheduled for Phase 3."
        )
