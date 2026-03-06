"""
ROI Extractor for rPPG Signal Acquisition
==========================================
Extracts forehead and cheek regions of interest (ROI) from video frames
using **MediaPipe FaceLandmarker** (Tasks API, compatible with
mediapipe >= 0.10.x).

On the first run the required FaceLandmarker model file (~6 MB) is
downloaded automatically to the ``models/`` directory adjacent to the
project root.

Motion stabilization is achieved via a partial-affine transform computed
from stable anatomical reference points (nose tip, eye inner corners, chin).
The frame is warped to a canonical pose before ROI sampling, reducing
motion-induced intensity fluctuations that would otherwise corrupt the
rPPG signal.

Reference:
    MediaPipe Tasks API  —
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python         # noqa: F401
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Model auto-download
# ---------------------------------------------------------------------------

_MODEL_DIR  = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_model() -> Path:
    """
    Return the path to the FaceLandmarker model, downloading it first if
    the file is not present in the local ``models/`` directory.
    """
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] FaceLandmarker model not found.\n"
            f"       Downloading from Google Storage -> {_MODEL_PATH}\n"
            f"       (one-time download, ~6 MB)"
        )
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model download complete.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark index groups
# All indices reference the 468-point canonical face mesh topology.
# ---------------------------------------------------------------------------

# Forehead region: forms a convex polygon above the eyebrows and below the
# approximate hairline. Selected to lie entirely on forehead skin, avoiding
# the hair, eyes, and temple regions.
FOREHEAD_INDICES: List[int] = [
    10, 338, 297, 332, 284, 251, 389,   # upper arc (near hairline)
    336, 296, 334, 293, 300,             # right eyebrow superior margin
    107,  66, 105,  63,  70,             # left eyebrow superior margin
    9, 8,                                # midline forehead
]

# Left cheek (camera-left / subject's right cheek):
# Bounded by the left zygomatic arch, lateral nose, and mandible.
LEFT_CHEEK_INDICES: List[int] = [
    234,  50,  36, 137, 123, 116, 117,
    147, 205, 206, 149, 150, 176,
    93, 132,  58, 172,
]

# Right cheek (camera-right / subject's left cheek):
# Mirror-symmetric counterpart of the left cheek region.
RIGHT_CHEEK_INDICES: List[int] = [
    454, 280, 266, 366, 352, 345, 346,
    376, 425, 426, 378, 379, 400,
    323, 361, 288, 397,
]

# Stable anatomical reference landmarks for affine motion stabilization.
# These points are geometrically rigid across expressions and head motion:
#   4   → nose tip
#   33  → left eye inner corner
#   263 → right eye inner corner
#   152 → chin (mentolabial sulcus)
STABLE_INDICES: List[int] = [4, 33, 263, 152]


class ROIExtractor:
    """
    Detects facial landmarks, stabilizes the frame, and extracts mean RGB
    signals from the combined forehead + cheek ROI.

    Public interface
    ----------------
    process(frame) → (rgb_mean | None, debug_frame)
        Single entry point. Runs MediaPipe once per frame.
    reset_reference()
        Clears the motion-stabilization reference (call on scene change).
    release()
        Frees MediaPipe resources.
    """

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Full application config dict. Reads the ``mediapipe`` block.
        """
        mp_cfg = config.get("mediapipe", {})

        model_path = _ensure_model()

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=mp_cfg.get("max_num_faces", 1),
            min_face_detection_confidence=float(
                mp_cfg.get("min_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                mp_cfg.get("min_tracking_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                mp_cfg.get("min_tracking_confidence", 0.5)
            ),
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Reference landmark positions for affine stabilization.
        # Set on the first successful detection; all subsequent frames
        # are warped to align with this reference configuration.
        self._reference_pts: Optional[np.ndarray] = None
        # Per-frame stable landmark positions for motion confidence estimation.
        self._prev_frame_pts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[dict], float, np.ndarray]:
        """
        Process one BGR frame: detect face → stabilize → extract per-ROI signals.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV, shape (H, W, 3).

        Returns
        -------
        roi_signals : dict or None
            Dictionary with keys ``"forehead"``, ``"left_cheek"``,
            ``"right_cheek"``.  Each value is an np.ndarray of shape (3,)
            containing the mean [R, G, B] over the HSV-filtered ROI, or
            ``None`` if that ROI had too few valid skin pixels after filtering.
            Returns ``None`` for the entire dict if no face is detected or
            all ROIs are empty.
        motion_confidence : float
            Score in [0, 1] based on frame-to-frame landmark displacement.
            1.0 = no motion; 0.0 = displacement ≥ threshold pixels.
            Callers should discard frames below their chosen threshold.
        debug_frame : np.ndarray
            BGR frame with ROI regions highlighted in green and a
            motion-confidence dot (green=good, red=high motion) in the
            top-right corner.  Returns a copy of the original if no face
            is detected.
        """
        h, w = frame.shape[:2]

        # FaceLandmarker expects an RGB mediapipe.Image
        rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_input)

        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, 1.0, frame.copy()

        # Use the first detected face (num_faces=1 by default)
        face_lms = result.face_landmarks[0]   # list of NormalizedLandmark
        lm_array = self._landmarks_to_array(face_lms, w, h)  # (468, 2)

        # ---- Motion confidence -------------------------------------------
        # Measure Euclidean displacement of rigid stable landmarks between
        # consecutive frames (pre-stabilisation image space).  Large
        # displacements indicate head movement that can corrupt the rPPG
        # signal even after affine compensation.
        current_stable_pts = lm_array[STABLE_INDICES].astype(np.float32)
        if self._prev_frame_pts is not None:
            _disp = float(np.mean(
                np.linalg.norm(current_stable_pts - self._prev_frame_pts, axis=1)
            ))
        else:
            _disp = 0.0
        self._prev_frame_pts = current_stable_pts.copy()
        _motion_threshold_px = 15.0
        motion_confidence = float(np.clip(1.0 - _disp / _motion_threshold_px, 0.0, 1.0))

        # Affine stabilization: warp frame to the canonical reference pose
        stabilized, M = self._stabilize(frame, lm_array, w, h)

        # Project all landmarks through the same affine transform
        stabilized_lm = self._transform_landmarks(lm_array, M)

        # Build per-ROI masks (with HSV skin filtering) on the stabilized frame
        roi_masks = self._build_per_roi_masks(stabilized_lm, stabilized, h, w)

        # Combined mask for debug overlay
        combined: Optional[np.ndarray] = None
        for _m in roi_masks.values():
            if _m is not None:
                combined = _m if combined is None else cv2.bitwise_or(combined, _m)

        # ---- Debug overlay ----
        debug_frame = stabilized.copy()
        if combined is not None and combined.sum() > 0:
            overlay = debug_frame.copy()
            overlay[combined > 0] = [0, 200, 0]
            cv2.addWeighted(overlay, 0.35, debug_frame, 0.65, 0, debug_frame)
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 1)

        # Motion-confidence dot: green = still, red = moving
        _dot_colour = (
            int(255 * (1.0 - motion_confidence)),
            int(255 * motion_confidence),
            0,
        )
        cv2.circle(debug_frame, (w - 20, 20), 8, _dot_colour, -1)

        # ---- Per-ROI mean RGB extraction ----
        rgb_stable = cv2.cvtColor(stabilized, cv2.COLOR_BGR2RGB)
        roi_signals: dict = {}
        for name, mask in roi_masks.items():
            if mask is not None and int(mask.sum()) >= 50:
                roi_signals[name] = self._masked_mean(rgb_stable, mask)
            else:
                roi_signals[name] = None

        if all(v is None for v in roi_signals.values()):
            return None, motion_confidence, debug_frame

        return roi_signals, motion_confidence, debug_frame

    def reset_reference(self) -> None:
        """
        Clear the stored stabilization reference frame.
        Call this if the camera is moved or the subject changes.
        """
        self._reference_pts  = None
        self._prev_frame_pts = None

    def release(self) -> None:
        """Release FaceLandmarker resources."""
        self.face_landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_array(landmarks, w: int, h: int) -> np.ndarray:
        """
        Convert normalized MediaPipe landmark coordinates to pixel space.

        Returns
        -------
        np.ndarray, shape (468, 2), dtype float32
        """
        return np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks],
            dtype=np.float32,
        )

    def _stabilize(
        self,
        frame: np.ndarray,
        lm_array: np.ndarray,
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate and apply a partial-affine (similarity) transform that
        maps the current stable landmark positions onto the stored reference
        configuration. This compensates for head translation, rotation, and
        scaling without introducing perspective distortion.

        On the first valid frame the current positions are stored as the
        reference and the identity transform is returned.

        Parameters
        ----------
        frame      : BGR image to warp.
        lm_array   : Current pixel-space landmarks, shape (468, 2).
        w, h       : Frame dimensions.

        Returns
        -------
        warped : Affine-warped BGR frame.
        M      : 2×3 affine matrix that was applied.
        """
        current_pts = lm_array[STABLE_INDICES].astype(np.float32)  # (4, 2)

        if self._reference_pts is None:
            # First detection: initialise reference → identity transform
            self._reference_pts = current_pts.copy()

        # estimateAffinePartial2D: 4 DOF (translation, rotation, uniform scale)
        # LMEDS is robust to occasional landmark jitter
        M, _ = cv2.estimateAffinePartial2D(
            current_pts, self._reference_pts, method=cv2.LMEDS
        )

        if M is None:
            # Estimation failed (too few inliers); fall back to identity
            M = np.eye(2, 3, dtype=np.float32)

        warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, M

    @staticmethod
    def _transform_landmarks(
        lm_array: np.ndarray, M: np.ndarray
    ) -> np.ndarray:
        """
        Apply a 2×3 affine matrix to all 468 landmark coordinates so that
        the ROI polygon aligns with the stabilized frame.

        Parameters
        ----------
        lm_array : (468, 2) float32
        M        : (2, 3) affine matrix

        Returns
        -------
        (468, 2) float32
        """
        ones = np.ones((lm_array.shape[0], 1), dtype=np.float32)
        lm_h = np.concatenate([lm_array, ones], axis=1)  # (468, 3)
        return (M @ lm_h.T).T                             # (468, 2)

    @staticmethod
    def _convex_hull_mask(pts: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Fill the convex hull of the given pixel points into a binary mask.

        Returns
        -------
        mask : uint8 array of shape (h, w), values ∈ {0, 1}
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        return mask

    def _build_roi_mask(
        self, lm_array: np.ndarray, h: int, w: int
    ) -> Optional[np.ndarray]:
        """
        Construct a combined binary mask covering the forehead and both
        cheek regions by taking the union of their convex hull masks.

        Returns None on any coordinate error (guards against edge-case
        landmark positions outside the frame boundary).
        """
        max_idx = max(
            max(FOREHEAD_INDICES),
            max(LEFT_CHEEK_INDICES),
            max(RIGHT_CHEEK_INDICES),
        )
        if len(lm_array) <= max_idx:
            return None

        try:
            forehead_mask = self._convex_hull_mask(
                lm_array[FOREHEAD_INDICES], h, w
            )
            left_mask = self._convex_hull_mask(
                lm_array[LEFT_CHEEK_INDICES], h, w
            )
            right_mask = self._convex_hull_mask(
                lm_array[RIGHT_CHEEK_INDICES], h, w
            )
            combined = cv2.bitwise_or(forehead_mask, left_mask)
            combined = cv2.bitwise_or(combined, right_mask)
            return combined
        except Exception:
            return None

    def _build_per_roi_masks(
        self,
        lm_array: np.ndarray,
        bgr_frame: np.ndarray,
        h: int,
        w: int,
    ) -> dict:
        """
        Build individual convex-hull masks for each ROI and apply an HSV
        skin filter to remove hair, shadows, specular highlights, and
        background contamination.

        Returns a dict with keys ``"forehead"``, ``"left_cheek"``,
        ``"right_cheek"``; values are uint8 masks (0/1) or ``None`` on
        coordinate error.

        Skin-filter fallback: if HSV filtering retains fewer than 20 % of
        the original anatomical mask the unfiltered mask is used instead,
        so signal extraction is never silently dropped under unusual lighting.
        """
        max_idx = max(
            max(FOREHEAD_INDICES),
            max(LEFT_CHEEK_INDICES),
            max(RIGHT_CHEEK_INDICES),
        )
        if len(lm_array) <= max_idx:
            return {"forehead": None, "left_cheek": None, "right_cheek": None}

        try:
            raw_masks = {
                "forehead":    self._convex_hull_mask(lm_array[FOREHEAD_INDICES],    h, w),
                "left_cheek":  self._convex_hull_mask(lm_array[LEFT_CHEEK_INDICES],  h, w),
                "right_cheek": self._convex_hull_mask(lm_array[RIGHT_CHEEK_INDICES], h, w),
            }
        except Exception:
            return {"forehead": None, "left_cheek": None, "right_cheek": None}

        result: dict = {}
        for name, raw in raw_masks.items():
            filtered = self._apply_skin_filter(bgr_frame, raw)
            # Require filtered mask to retain ≥ 20% of original ROI area
            if int(filtered.sum()) >= max(50, int(0.20 * max(1, int(raw.sum())))):
                result[name] = filtered
            else:
                result[name] = raw   # fallback: accept unfiltered anatomical ROI
        return result

    @staticmethod
    def _apply_skin_filter(bgr_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refine a binary ROI mask using HSV colour filtering to retain only
        skin-like pixels, rejecting hair, shadows, specular reflections,
        and achromatic background regions.

        Two HSV clusters are unioned to cover the full skin-tone gamut
        including the red wrap-around (H ≈ 170–180 in OpenCV's half-angle
        convention):

        * Cluster 1: H ∈ [0, 20],   S ∈ [35, 165], V ∈ [60, 230]
        * Cluster 2: H ∈ [170, 180], S ∈ [35, 165], V ∈ [60, 230]

        Parameters
        ----------
        bgr_frame : np.ndarray, shape (H, W, 3), dtype uint8
            Stabilized BGR frame.
        mask : np.ndarray, shape (H, W), dtype uint8  (values 0 or 1)
            Anatomical ROI convex-hull mask to refine.

        Returns
        -------
        np.ndarray, shape (H, W), dtype uint8
            Refined mask with 0/1 values.
        """
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        skin1 = cv2.inRange(
            hsv,
            np.array([0,   35,  60], dtype=np.uint8),
            np.array([20,  165, 230], dtype=np.uint8),
        )
        skin2 = cv2.inRange(
            hsv,
            np.array([170, 35,  60], dtype=np.uint8),
            np.array([180, 165, 230], dtype=np.uint8),
        )
        skin_hsv    = cv2.bitwise_or(skin1, skin2)         # 0 or 255
        skin_binary = (skin_hsv > 0).astype(np.uint8)      # 0 or 1
        return cv2.bitwise_and(skin_binary, mask)

    @staticmethod
    def _masked_mean(rgb_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Compute the spatial mean RGB value over all pixels within the mask.

        Parameters
        ----------
        rgb_frame : (H, W, 3) uint8 array in RGB order
        mask      : (H, W) uint8 binary mask

        Returns
        -------
        np.ndarray of shape (3,), dtype float64 — mean [R, G, B]
        """
        pixels = rgb_frame[mask > 0]   # (N, 3)
        return pixels.mean(axis=0).astype(np.float64)
