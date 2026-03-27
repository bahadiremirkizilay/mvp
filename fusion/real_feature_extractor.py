"""Real video feature extraction for deception dataset samples.

This module builds modality-aligned temporal features directly from video frames.
It intentionally avoids label-dependent synthesis used by smoke fallback mode.

Literature-backed improvements (Yang et al. 2022, Mathur & Matarić 2020, Ding CVPR 2019):
  - Audio MFCC features via librosa (all top papers validate audio as critical modality)
  - Verbal/linguistic features from transcripts (Pérez-Rosas 2015, Rill-Garcia 2019)
  - Face ROI detection for less-noisy rPPG signal (Ding CVPR 2019)
  - CNN emotion features from CASME II pre-trained ResNet-18 (per-frame softmax probs)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class RealVideoFeatureExtractor:
    """Extract lightweight temporal features from raw video frames."""

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        target_length: int = 20,
        max_video_frames: int = 180,
        use_gesture_summary: bool = True,
        emotion_model_path: Optional[str] = None,
    ):
        self.modalities = modalities or ["rppg", "emotion", "behavioral"]
        self.target_length = int(target_length)
        self.max_video_frames = int(max_video_frames)
        self.use_gesture_summary = bool(use_gesture_summary)

        # CNN emotion model (lazily loaded from checkpoint)
        self._emotion_model = None
        self._emotion_num_classes = 16   # fallback pixel-stats dim

        # Reported output dims — used by dataset to align default_dims
        self.output_dims: Dict[str, int] = {
            "emotion": self._emotion_num_classes,
        }

        if emotion_model_path and Path(emotion_model_path).exists():
            self._load_emotion_model(emotion_model_path)

    def _load_emotion_model(self, path: str) -> None:
        """Load CASME II pre-trained ResNet from checkpoint."""
        try:
            import torch
            from torchvision import models

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            backbone = ckpt.get("backbone", "resnet18")
            num_classes = ckpt.get("num_classes", 7)

            if backbone == "resnet18":
                model = models.resnet18(weights=None)
            else:
                model = models.resnet50(weights=None)

            import torch.nn as nn
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
            model.load_state_dict(ckpt["model_state_dict"])

            self._emotion_device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(self._emotion_device)
            model.eval()

            self._emotion_model = model
            self._emotion_num_classes = num_classes
            self.output_dims["emotion"] = num_classes
            print(f"[Extractor] Loaded CASME II emotion model ({backbone}, {num_classes} classes) → {self._emotion_device}")
        except Exception as exc:
            print(f"[Extractor] WARNING: Could not load emotion model: {exc}")
            self._emotion_model = None

    def __call__(self, video_path: str, audio_path: Optional[str] = None, row=None) -> Dict[str, np.ndarray]:
        frames = self._decode_frames(video_path)
        if frames is None or frames.size == 0:
            return {}

        out = self._build_modalities(frames, row=row)

        # Audio features: extract from video audio track via ffmpeg + librosa
        if "audio" in self.modalities:
            audio_feats = self._extract_audio_mfcc(video_path)
            if audio_feats is not None:
                out["audio"] = audio_feats

        # Verbal features: linguistic statistics from transcript
        if "verbal" in self.modalities:
            verbal_feats = self._extract_verbal_features(row)
            if verbal_feats is not None:
                out["verbal"] = verbal_feats

        return out

    def _decode_frames(self, video_path: str) -> Optional[np.ndarray]:
        path = Path(video_path)
        if not path.exists():
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        sampled: List[np.ndarray] = []
        frames_read = 0

        while frames_read < self.max_video_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                continue

            frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            sampled.append(frame)
            frames_read += 1

        cap.release()

        if not sampled:
            return None

        x = np.stack(sampled, axis=0).astype(np.float32) / 255.0
        return self._resample_time(x, self.target_length)

    def _build_modalities(self, frames_bgr: np.ndarray, row=None) -> Dict[str, np.ndarray]:
        # Convert once and reuse to keep extraction fast.
        rgb = frames_bgr[..., ::-1]
        gray = np.stack([cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for f in frames_bgr], axis=0)
        gray = gray.astype(np.float32) / 255.0

        # Detect face ROI in middle frame for better rPPG signal (Ding CVPR 2019)
        face_rgb = self._extract_face_roi(frames_bgr, rgb)

        mean_rgb = face_rgb.mean(axis=(1, 2))
        std_rgb = face_rgb.std(axis=(1, 2))

        motion = np.zeros((gray.shape[0],), dtype=np.float32)
        if gray.shape[0] > 1:
            motion[1:] = np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))

        sat = (rgb.max(axis=3) - rgb.min(axis=3)).mean(axis=(1, 2))
        val = rgb.max(axis=3).mean(axis=(1, 2))
        contrast = gray.std(axis=(1, 2))

        gx = np.stack([cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) for g in gray], axis=0)
        gy = np.stack([cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) for g in gray], axis=0)
        edge = np.sqrt(gx * gx + gy * gy).mean(axis=(1, 2))
        horiz = np.mean(np.abs(gx), axis=(1, 2))
        vert = np.mean(np.abs(gy), axis=(1, 2))

        # POS-like pulse proxy from face ROI channel differences (better SNR).
        r = mean_rgb[:, 0]
        g = mean_rgb[:, 1]
        b = mean_rgb[:, 2]
        s1 = g - b
        s2 = -2.0 * r + g + b
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        pulse = s1 + alpha * s2
        pulse = (pulse - pulse.mean()) / (pulse.std() + 1e-6)

        out: Dict[str, np.ndarray] = {}
        g_mean, g_std, g_count = self._gesture_summary(row)

        if "rppg" in self.modalities:
            out["rppg"] = np.stack(
                [
                    r,
                    g,
                    b,
                    pulse,
                    g / (r + 1e-6),
                    r - g,
                    g - b,
                    motion,
                ],
                axis=1,
            ).astype(np.float32)
            out["rppg"] = self._normalize_sequence(out["rppg"])

        if "emotion" in self.modalities:
            if self._emotion_model is not None:
                # Use CASME II pre-trained CNN: per-frame softmax probabilities (T, 7)
                cnn_em = self._extract_emotion_cnn(frames_bgr)
                if cnn_em is not None:
                    out["emotion"] = cnn_em   # already normalized (probs sum to 1)
            if "emotion" not in out:
                # Fallback: pixel-level statistics (16-dim) when no model is loaded
                em = np.stack(
                    [
                        val,
                        contrast,
                        sat,
                        edge,
                        r,
                        g,
                        b,
                        motion,
                        np.gradient(val),
                        np.gradient(contrast),
                        np.gradient(sat),
                        np.gradient(edge),
                        np.gradient(r),
                        np.gradient(g),
                        np.gradient(b),
                        np.gradient(motion),
                    ],
                    axis=1,
                ).astype(np.float32)
                out["emotion"] = self._normalize_sequence(em)

        if "behavioral" in self.modalities:
            # Try MediaPipe face-mesh behavioral features first (richer, more generalizable)
            mp_beh = self._extract_behavioral_mediapipe(frames_bgr)
            if mp_beh is not None:
                out["behavioral"] = mp_beh
            else:
                # Fallback: pixel-level statistics
                beh = np.stack(
                    [
                        motion,
                        np.gradient(motion),
                        edge,
                        horiz,
                        vert,
                        contrast,
                        np.full_like(motion, g_mean),
                        np.full_like(motion, g_std),
                        np.full_like(motion, g_count),
                    ],
                    axis=1,
                ).astype(np.float32)
                out["behavioral"] = self._normalize_sequence(beh)

        return out

    def _extract_behavioral_mediapipe(self, frames_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract per-frame behavioral signals via MediaPipe Face Landmarker (Tasks API).

        Returns shape (T, 9) — z-score normalised per video:
          0: head_yaw    — left/right turn from 3-D transformation matrix
          1: head_pitch  — up/down tilt from 3-D transformation matrix
          2: head_roll   — lateral tilt from 3-D transformation matrix
          3: eye_blink   — mean blink blendshape (eyeBlinkLeft+Right / 2)
          4: eye_squint  — mean squint blendshape (eyeSquintLeft+Right / 2)
          5: jaw_open    — mouth opening blendshape
          6: mouth_smile — mean smile blendshape (mouthSmileLeft+Right / 2)
          7: brow_up     — inner brow raise (stress / surprise)
          8: face_motion — inter-frame nose displacement (kinematic fidgeting)

        Uses models/face_landmarker.task for FACS blendshapes + 3-D head pose.
        All features are person- and position-invariant — no pixel statistics.
        """
        try:
            import os
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models", "face_landmarker.task",
            )
            if not os.path.isfile(model_path):
                return None

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.2,
                min_face_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
            detector = mp_vision.FaceLandmarker.create_from_options(opts)

            rows: list = []
            prev_nose_xy: Optional[np.ndarray] = None

            for frame in frames_bgr:
                frame_u8  = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
                frame_rgb = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2RGB)
                mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result    = detector.detect(mp_img)

                if result.face_landmarks and result.face_blendshapes:
                    lm = result.face_landmarks[0]
                    bs = {b.category_name: b.score for b in result.face_blendshapes[0]}

                    # Head pose: real 3-D rotation from the transformation matrix
                    if result.facial_transformation_matrixes:
                        R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
                        pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                        yaw   = float(np.arctan2(R[1, 0], R[0, 0]))
                        roll  = float(np.arctan2(R[2, 1], R[2, 2]))
                    else:
                        # Landmark proxy fallback (no calibration needed)
                        xs = [l.x for l in lm]; ys = [l.y for l in lm]
                        cx = (max(xs) + min(xs)) / 2
                        cy = (max(ys) + min(ys)) / 2
                        fw = max(xs) - min(xs) + 1e-6
                        fh = max(ys) - min(ys) + 1e-6
                        yaw   = (lm[4].x - cx) / fw
                        pitch = (lm[4].y - cy) / fh
                        roll  = 0.0

                    # FACS blendshape features (deception-relevant action units)
                    eye_blink   = (bs.get("eyeBlinkLeft",   0.0) + bs.get("eyeBlinkRight",   0.0)) / 2.0
                    eye_squint  = (bs.get("eyeSquintLeft",  0.0) + bs.get("eyeSquintRight",  0.0)) / 2.0
                    jaw_open    =  bs.get("jawOpen",        0.0)
                    mouth_smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
                    brow_up     =  bs.get("browInnerUp",    0.0)

                    # Kinematic: nose displacement across consecutive frames
                    nose_xy   = np.array([lm[4].x, lm[4].y])
                    face_w    = max([l.x for l in lm]) - min([l.x for l in lm]) + 1e-6
                    face_mot  = (float(np.linalg.norm(nose_xy - prev_nose_xy) / face_w)
                                 if prev_nose_xy is not None else 0.0)
                    prev_nose_xy = nose_xy

                    rows.append([yaw, pitch, roll,
                                 eye_blink, eye_squint, jaw_open,
                                 mouth_smile, brow_up, face_mot])
                else:
                    rows.append(rows[-1] if rows else [0.0] * 9)
                    prev_nose_xy = None

            detector.close()

            arr = np.array(rows, dtype=np.float32)   # (T, 9)
            if arr.std() < 1e-8:                     # all-constant → no signal
                return None
            return self._normalize_sequence(arr)

        except Exception:
            return None

    def _resample_time(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        if arr.shape[0] == target_length:
            return arr
        if arr.shape[0] == 1:
            return np.repeat(arr, target_length, axis=0)

        src_t = np.linspace(0.0, 1.0, num=arr.shape[0], dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
        flat = arr.reshape(arr.shape[0], -1)
        cols = [np.interp(dst_t, src_t, flat[:, j]).astype(np.float32) for j in range(flat.shape[1])]
        out = np.stack(cols, axis=1)
        return out.reshape(target_length, *arr.shape[1:])

    def _normalize_sequence(self, x: np.ndarray) -> np.ndarray:
        """Per-feature temporal standardization to reduce subject/lighting bias."""
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        return ((x - mu) / (sd + 1e-6)).astype(np.float32)

    def _extract_emotion_cnn(self, frames_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run CASME II pre-trained ResNet on each frame to get per-frame emotion probs.

        Returns shape (T, num_classes) — e.g. (20, 7) for CASME II 7-class model.
        Each row is a softmax probability distribution over emotion categories.

        This is far more generalizable than pixel statistics:
        - Task-relevant features (expression change) rather than lighting/exposure
        - Trained on diverse labeled faces, not on the target dataset
        - Position-invariant: same expression at any time = same features
        """
        try:
            import torch
            from torchvision import transforms as T

            preprocess = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            device = getattr(self, "_emotion_device", "cpu")

            # Build batch of face crops
            crops = []
            for frame in frames_bgr:
                frame_u8  = (frame * 255).astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2RGB)

                # Try to crop to face region for cleaner expression features
                gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    pad = int(0.1 * min(w, h))
                    x1 = max(0, x - pad); y1 = max(0, y - pad)
                    x2 = min(frame_rgb.shape[1], x + w + pad)
                    y2 = min(frame_rgb.shape[0], y + h + pad)
                    crop = frame_rgb[y1:y2, x1:x2]
                else:
                    crop = frame_rgb
                crops.append(preprocess(crop))

            # Single batched forward pass
            batch = torch.stack(crops, dim=0).to(device)   # (T, 3, 224, 224)
            with torch.no_grad():
                logits = self._emotion_model(batch)         # (T, num_classes)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()

            return probs.astype(np.float32)  # (T, num_classes)

        except Exception:
            return None

    def _extract_face_roi(self, frames_bgr: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Detect face in middle frame and use ROI for all frames (better rPPG SNR).

        Falls back to full-frame RGB if no face is detected.
        """
        # Try face detection on the middle frame
        mid = len(frames_bgr) // 2
        frame_u8 = (frames_bgr[mid] * 255).astype(np.uint8)
        gray_mid = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray_mid, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )
        except Exception:
            faces = []

        if len(faces) == 0:
            return rgb  # fallback: use full frame

        # Use largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        h_frame, w_frame = frames_bgr.shape[1], frames_bgr.shape[2]

        # Expand box 10% for forehead context
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_frame, x + w + pad_x)
        y2 = min(h_frame, y + h + pad_y)

        face_roi = rgb[:, y1:y2, x1:x2, :]
        if face_roi.size == 0:
            return rgb
        return face_roi

    def _extract_audio_mfcc(self, video_path: str) -> Optional[np.ndarray]:
        """Extract MFCC + delta features from video audio track.

        Uses ffmpeg to decode audio, then librosa to compute MFCCs.
        Returns shape (target_length, 24): 12 MFCCs + 12 delta-MFCCs.

        Based on literature: Yang et al. 2022 (92.78% acc) uses audio emotional
        states + openSMILE; Rill Garcia 2019 uses acoustic features; all top
        deception papers include audio as a critical modality.
        """
        try:
            import librosa
        except ImportError:
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ac", "1", "-ar", "16000", "-vn",
                    tmp_path,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0 or not os.path.exists(tmp_path):
                return None

            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            if len(y) < sr * 0.1:  # shorter than 100ms
                return None

            # Compute 12 MFCCs per frame
            hop_length = max(1, len(y) // (self.target_length * 4))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=hop_length)
            delta = librosa.feature.delta(mfcc)
            features = np.vstack([mfcc, delta]).T.astype(np.float32)  # (T_audio, 24)

            return self._normalize_sequence(
                self._resample_time(features, self.target_length)
            )

        except Exception:
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _extract_verbal_features(self, row) -> Optional[np.ndarray]:
        """Extract linguistic statistics from transcription file.

        Returns shape (target_length, 12): clip-level features replicated over T.

        Based on: Pérez-Rosas & Mihalcea 2015 (linguistic features strongest single
        modality), Rill-Garcia 2019 (multimodal + text best combination).
        Feature set: word rate, lexical diversity, filler/negation/self-ref/certainty
        word rates, avg word length, question/exclamation marks, char rate.
        """
        if row is None:
            return None

        txt_path = str(row.get("transcription_path", "") or "")
        if not txt_path:
            return None

        try:
            text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        text_lower = text.lower()
        words = text_lower.split()
        n = max(1, len(words))
        duration = float(row.get("duration_sec", 1.0) or 1.0)

        filler_words = {"um", "uh", "er", "hmm", "like", "you", "know", "basically", "literally", "actually"}
        negative_words = {
            "not", "no", "never", "nobody", "nothing", "neither", "nor",
            "deny", "denied", "dont", "didnt", "cant", "wont", "wouldnt",
            "shouldnt", "couldnt", "wasnt", "werent", "hadnt", "havent",
        }
        self_ref = {"i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"}
        certainty_words = {
            "definitely", "certainly", "absolutely", "clearly", "obviously",
            "always", "never", "sure", "positive", "exactly", "truly",
        }
        cognitive_words = {
            "think", "know", "believe", "remember", "understand", "realize",
            "suppose", "assume", "feel", "thought", "guess", "wonder",
        }

        # Normalize apostrophes for matching
        words_clean = [w.replace("'", "").replace('"', "") for w in words]

        feats = np.array(
            [
                n / max(1.0, duration),                                          # word rate (words/sec)
                len(set(words_clean)) / n,                                       # lexical diversity (TTR)
                sum(w in filler_words for w in words_clean) / n,                 # filler word rate
                sum(w in negative_words for w in words_clean) / n,               # negation rate
                sum(w in self_ref for w in words_clean) / n,                     # self-reference rate
                sum(w in certainty_words for w in words_clean) / n,              # certainty rate
                sum(w in cognitive_words for w in words_clean) / n,              # cognitive complexity rate
                float(np.mean([len(w) for w in words_clean])) / 8.0,             # avg word length (normalized)
                float(text.count("?")) / n,                                      # question rate
                float(text.count("!")) / n,                                      # exclamation rate
                len(text) / max(1.0, duration) / 100.0,                          # char rate (normalized)
                float(int(bool(row.get("has_transcription", False)))),           # has transcript flag
            ],
            dtype=np.float32,
        )

        # Replicate over time axis (clip-level feature, no temporal variation)
        return np.tile(feats[None, :], (self.target_length, 1)).astype(np.float32)

    def _gesture_summary(self, row) -> tuple[float, float, float]:
        if not self.use_gesture_summary or row is None:
            return 0.0, 0.0, 0.0

        vals = []
        skip = {
            "video_id", "video_path", "label", "label_name", "duration_sec", "fps", "num_frames",
            "has_transcription", "transcription_path", "quality_flags", "class"
        }
        try:
            for col, raw in row.items():
                if str(col) in skip:
                    continue
                try:
                    v = float(raw)
                    if v in (0.0, 1.0):
                        vals.append(v)
                except Exception:
                    continue
        except Exception:
            return 0.0, 0.0, 0.0

        if not vals:
            return 0.0, 0.0, 0.0

        a = np.asarray(vals, dtype=np.float32)
        return float(a.mean()), float(a.std()), float(a.sum())
