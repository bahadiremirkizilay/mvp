"""
Gaze Estimation  [Phase 3 — Not yet implemented]
=================================================
Will estimate gaze direction (yaw/pitch angles) and pupil position
relative to the iris using MediaPipe Iris landmarks (refined mesh).

Planned outputs:
    • gaze_vector   : (3,) unit vector in camera space
    • fixation_x/y  : normalised screen-plane gaze coordinates
    • saccade_flag  : bool — rapid eye movement detected
    • avg_pupil_diam : float — relative pupil diameter (cognitive arousal proxy)
"""

# TODO (Phase 3): Implement GazeEstimator


class GazeEstimator:
    """Placeholder — implement in Phase 3."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GazeEstimator is scheduled for Phase 3."
        )
