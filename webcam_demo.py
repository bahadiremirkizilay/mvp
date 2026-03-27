#!/usr/bin/env python3
"""
Webcam Real-Time Deception Detection Demo
==========================================
Kameradan canlı görüntü alarak behavioral features çıkarır,
eğitilmiş LOSO modelini çalıştırır ve ekranda "lie risk" skoru gösterir.

Kullanım:
    python webcam_demo.py
    python webcam_demo.py --model checkpoints/deception_loso_v5_behavioral/fold_115/best_fusion_model.pth
    python webcam_demo.py --camera 1          # farklı kamera index
    python webcam_demo.py --window_secs 5     # kaç saniyelik pencere (default=4)
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
import torch

# ── Proje root'unu path'e ekle ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from fusion.fusion_model import FusionModel

# ── Renkler (BGR) ───────────────────────────────────────────────────────────
CLR_GREEN  = (56, 200, 56)
CLR_YELLOW = (0, 200, 220)
CLR_RED    = (40, 40, 220)
CLR_GRAY   = (160, 160, 160)
CLR_WHITE  = (240, 240, 240)
CLR_BLACK  = (0, 0, 0)
CLR_BG     = (30, 30, 30)


# ────────────────────────────────────────────────────────────────────────────
# MediaPipe behavioral extractor (aynen real_feature_extractor'dan alındı)
# ────────────────────────────────────────────────────────────────────────────
class OnlineBehavioralExtractor:
    """Her frame'den 9-dim behavioral feature çıkarır (MediaPipe Face Landmarker)."""

    FEATURE_NAMES = [
        "head_yaw", "head_pitch", "head_roll",
        "eye_blink", "eye_squint", "jaw_open",
        "mouth_smile", "brow_up", "face_motion",
    ]
    DIM = 9

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._prev_nose: Optional[np.ndarray] = None

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Single frame → (9,) float32, or None if no face detected."""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            self._prev_nose = None
            return None

        # ── Head pose from 3-D transformation matrix ──
        mat = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        yaw   = float(np.arctan2(-mat[2, 0], sy))
        pitch = float(np.arctan2(mat[2, 1], mat[2, 2]))
        roll  = float(np.arctan2(mat[1, 0], mat[0, 0]))

        # ── Blendshapes ──
        bs = {b.category_name: float(b.score)
              for b in result.face_blendshapes[0]} if result.face_blendshapes else {}

        eye_blink  = (bs.get("eyeBlinkLeft",  0.0) + bs.get("eyeBlinkRight",  0.0)) / 2.0
        eye_squint = (bs.get("eyeSquintLeft", 0.0) + bs.get("eyeSquintRight", 0.0)) / 2.0
        jaw_open   = bs.get("jawOpen", 0.0)
        mouth_smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        brow_up    = (bs.get("browInnerUp", 0.0) + bs.get("browOuterUpLeft", 0.0)) / 2.0

        # ── Face motion (nose displacement) ──
        lm = result.face_landmarks[0]
        nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
        if self._prev_nose is not None:
            face_motion = float(np.linalg.norm(nose - self._prev_nose))
        else:
            face_motion = 0.0
        self._prev_nose = nose

        return np.array([yaw, pitch, roll, eye_blink, eye_squint,
                         jaw_open, mouth_smile, brow_up, face_motion],
                        dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Model yükleme
# ────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str) -> FusionModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoint içinde config var mı?
    cfg = ckpt.get("config", {})
    hidden_dim  = cfg.get("hidden_dim",  96)
    num_layers  = cfg.get("num_layers",  1)
    num_heads   = cfg.get("num_heads",   4)
    dropout     = cfg.get("dropout",     0.3)

    model = FusionModel(
        input_dims={"behavioral": 9},
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ────────────────────────────────────────────────────────────────────────────
# Z-score normalize (per-sequence, aynen training sırasındaki gibi)
# ────────────────────────────────────────────────────────────────────────────
def normalize_seq(seq: np.ndarray) -> np.ndarray:
    mu  = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mu) / std


def resample_to_T(seq: np.ndarray, T: int = 20) -> np.ndarray:
    """Temporal resample to exactly T timesteps."""
    n = len(seq)
    if n == T:
        return seq
    idx = np.linspace(0, n - 1, T).astype(int)
    return seq[idx]


# ────────────────────────────────────────────────────────────────────────────
# HUD overlay yardımcıları
# ────────────────────────────────────────────────────────────────────────────
def draw_gauge(frame: np.ndarray, score: float, cx: int, cy: int, r: int = 60):
    """Yarım daire gauge çiz."""
    # Arka plan yayı
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, 360, (50, 50, 50), 8)
    # Değer yayı
    angle = int(180 + score * 180)
    color = CLR_GREEN if score < 0.4 else (CLR_YELLOW if score < 0.65 else CLR_RED)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, angle, color, 8)
    # Merkez nokta
    cv2.circle(frame, (cx, cy), 5, CLR_WHITE, -1)


def draw_bar(frame: np.ndarray, label: str, value: float,
             x: int, y: int, w: int = 180, h: int = 18):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill_w = int(w * max(0.0, min(1.0, value)))
    color = CLR_GREEN if value < 0.4 else (CLR_YELLOW if value < 0.65 else CLR_RED)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_GRAY, 1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x + 4, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)


def draw_feature_bars(frame: np.ndarray, feat_names: list, feat_vals: np.ndarray,
                      x: int, y: int):
    for i, (name, val) in enumerate(zip(feat_names, feat_vals)):
        # Normalize 0-1 for display (tanh scale)
        disp = float(0.5 + 0.5 * np.tanh(val))
        draw_bar(frame, name[:10], disp, x, y + i * 22, w=160, h=16)


# ────────────────────────────────────────────────────────────────────────────
# Ana demo döngüsü
# ────────────────────────────────────────────────────────────────────────────
def run_demo(
    checkpoint: str,
    camera_idx: int = 0,
    window_secs: float = 4.0,
    target_T: int = 20,
    infer_every_secs: float = 0.5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demo] Device: {device}")

    # Model
    print(f"[Demo] Loading model: {checkpoint}")
    model = load_model(checkpoint, device)
    print("[Demo] Model loaded OK")

    # MediaPipe
    mp_model = Path(__file__).parent / "models" / "face_landmarker.task"
    if not mp_model.exists():
        print(f"[ERROR] face_landmarker.task not found at {mp_model}")
        print("  → Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        sys.exit(1)

    print("[Demo] Loading MediaPipe Face Landmarker …")
    extractor = OnlineBehavioralExtractor(str(mp_model))
    print("[Demo] MediaPipe ready")

    # Kamera
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Camera {camera_idx} could not be opened.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    buffer_size = max(target_T, int(fps * window_secs))

    feat_buffer: Deque[np.ndarray] = collections.deque(maxlen=buffer_size)

    lie_risk     = 0.0
    stress_level = 0.0
    last_infer   = 0.0
    last_feat    = np.zeros(OnlineBehavioralExtractor.DIM, dtype=np.float32)

    # Geçmiş skor grafiği
    history_len = 80
    risk_history: Deque[float] = collections.deque([0.0] * history_len, maxlen=history_len)

    print("[Demo] Running. Press Q to quit.")
    cv2.namedWindow("Deception Detection Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deception Detection Demo", 900, 520)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror
        display = np.zeros((520, 900, 3), dtype=np.uint8)
        display[:] = CLR_BG

        # ── Feature extraction ──────────────────────────────────────────────
        small = cv2.resize(frame, (320, 240))
        feat = extractor.extract(small)
        if feat is not None:
            feat_buffer.append(feat)
            last_feat = feat

        # ── Inference (infer_every_secs'de bir) ─────────────────────────────
        now = time.time()
        if (now - last_infer) >= infer_every_secs and len(feat_buffer) >= target_T:
            seq = np.array(feat_buffer, dtype=np.float32)
            seq = resample_to_T(seq, target_T)
            seq = normalize_seq(seq)

            with torch.no_grad():
                x = torch.from_numpy(seq[None]).float().to(device)  # [1, T, 9]
                out = model({"behavioral": x})
                lie_risk     = float(out["lie_risk"].item())
                stress_level = float(out["stress_level"].item())

            risk_history.append(lie_risk)
            last_infer = now

        # ── Kamera görüntüsü (sol panel) ─────────────────────────────────────
        cam_h, cam_w = 360, 480
        cam_disp = cv2.resize(frame, (cam_w, cam_h))
        display[80:80 + cam_h, 10:10 + cam_w] = cam_disp

        # Yüz algılandı mı?
        face_ok = feat is not None
        face_label = "Face: OK" if face_ok else "Face: NOT DETECTED"
        face_color = CLR_GREEN if face_ok else CLR_RED
        cv2.putText(display, face_label, (15, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, face_color, 1, cv2.LINE_AA)

        # ── Başlık ───────────────────────────────────────────────────────────
        cv2.putText(display, "DECEPTION DETECTION DEMO", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(display, f"Buffer: {len(feat_buffer)}/{buffer_size}  |  Model: behavioral+v5",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_GRAY, 1, cv2.LINE_AA)

        # ── Sağ panel ─────────────────────────────────────────────────────────
        rx = 510  # sağ panel başlangıç x

        # Ana skor etiketi
        risk_pct = int(lie_risk * 100)
        if lie_risk < 0.4:
            verdict, vclr = "TRUTH", CLR_GREEN
        elif lie_risk < 0.65:
            verdict, vclr = "UNCERTAIN", CLR_YELLOW
        else:
            verdict, vclr = "DECEPTIVE", CLR_RED

        cv2.putText(display, verdict, (rx, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, vclr, 2, cv2.LINE_AA)
        cv2.putText(display, f"Lie Risk: {risk_pct}%", (rx, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, vclr, 1, cv2.LINE_AA)

        # Gauge
        draw_gauge(display, lie_risk, cx=rx + 80, cy=200, r=65)
        cv2.putText(display, f"{risk_pct}%", (rx + 65, 195),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, vclr, 2, cv2.LINE_AA)

        # Stress bar
        draw_bar(display, "Stress", stress_level, rx, 280, w=200, h=22)

        # Feature bar'ları
        cv2.putText(display, "Live Features:", (rx, 325),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GRAY, 1, cv2.LINE_AA)
        draw_feature_bars(display, OnlineBehavioralExtractor.FEATURE_NAMES,
                          last_feat, rx, 335)

        # ── Risk geçmişi grafiği ──────────────────────────────────────────────
        gh = 60
        gw = 370
        gx, gy = 10, 450
        # Arkaplan
        cv2.rectangle(display, (gx, gy), (gx + gw, gy + gh), (40, 40, 40), -1)
        cv2.putText(display, "Risk History", (gx + 2, gy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_GRAY, 1, cv2.LINE_AA)
        # 0.5 orta çizgisi
        mid_y = gy + gh // 2
        cv2.line(display, (gx, mid_y), (gx + gw, mid_y), (70, 70, 70), 1)
        # Grafik
        hist = list(risk_history)
        pts = []
        for i, v in enumerate(hist):
            px = gx + int(i * gw / history_len)
            py = gy + gh - int(v * gh)
            pts.append((px, py))
        if len(pts) > 1:
            for i in range(1, len(pts)):
                v = hist[i]
                c = CLR_GREEN if v < 0.4 else (CLR_YELLOW if v < 0.65 else CLR_RED)
                cv2.line(display, pts[i - 1], pts[i], c, 2)

        # ── Göster ──────────────────────────────────────────────────────────
        cv2.imshow("Deception Detection Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Demo] Finished.")


# ────────────────────────────────────────────────────────────────────────────
def main():
    default_ckpt = str(
        Path(__file__).parent
        / "checkpoints/deception_loso_v5_behavioral/fold_115/best_fusion_model.pth"
    )

    parser = argparse.ArgumentParser(description="Webcam deception detection demo")
    parser.add_argument("--model",        default=default_ckpt,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--camera",       type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--window_secs",  type=float, default=4.0,
                        help="Rolling buffer duration in seconds (default: 4)")
    parser.add_argument("--infer_every",  type=float, default=0.5,
                        help="Inference interval in seconds (default: 0.5)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("Hint: En iyi fold checkpoint'i: "
              "checkpoints/deception_loso_v5_behavioral/fold_115/best_fusion_model.pth")
        sys.exit(1)

    run_demo(
        checkpoint=args.model,
        camera_idx=args.camera,
        window_secs=args.window_secs,
        infer_every_secs=args.infer_every,
    )


if __name__ == "__main__":
    main()
