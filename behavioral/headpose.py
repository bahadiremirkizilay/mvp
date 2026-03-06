"""
Head Pose Estimation  [Phase 3 — Not yet implemented]
======================================================
Will recover 3-D head orientation (roll, pitch, yaw) by solving the
PnP problem between canonical 3-D model landmarks and detected 2-D
MediaPipe Face Mesh landmarks using cv2.solvePnP (EPNP algorithm).

Planned outputs:
    • euler_angles : (roll, pitch, yaw) in degrees
    • rotation_vec : Rodrigues rotation vector
    • translation_vec : camera-space translation
    • attention_flag  : bool — head pointing toward camera within ±15°

Reference:
    Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment
    with an ensemble of regression trees. CVPR.
"""

# TODO (Phase 3): Implement HeadPoseEstimator


class HeadPoseEstimator:
    """Placeholder — implement in Phase 3."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HeadPoseEstimator is scheduled for Phase 3."
        )
