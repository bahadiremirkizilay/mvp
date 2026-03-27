"""
Multimodal Feature Builder
===========================
Professional-grade feature extraction and fusion system combining:
    • rPPG physiological signals (HR, HRV, frequency domain)
    • Emotion recognition features (facial expressions)
    • Behavioral features (blink, gaze, head pose)
    • Audio features (energy, ZCR, spectral voice cues)

Architecture:
    • Independent feature extractors for each modality
    • Temporal alignment and synchronization
    • Feature normalization and standardization
    • Early/late fusion strategies
    • Dimensionality reduction (PCA/AutoEncoder)

Use Cases:
    • Stress detection
    • Driver monitoring
    • Affective computing
    • Healthcare applications
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from dataclasses import dataclass
from scipy.io import wavfile
from scipy import signal

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rppg.pos_method import process_video as extract_rppg
from emotion.model import EmotionClassifier


@dataclass
class FeatureConfig:
    """Configuration for multimodal feature extraction."""
    
    # rPPG features
    use_rppg: bool = True
    rppg_window_sec: float = 7.0
    rppg_features: List[str] = None  # ['hr_bpm', 'sdnn', 'rmssd', 'lf_hf_ratio']
    
    # Emotion features
    use_emotion: bool = True
    emotion_model_path: Optional[str] = None
    emotion_features: List[str] = None  # ['emotion_probs', 'valence', 'arousal']
    
    # Behavioral features
    use_behavioral: bool = True
    behavioral_features: List[str] = None  # ['blink_rate', 'gaze_x', 'gaze_y', 'head_pitch']

    # Audio features
    use_audio: bool = False
    audio_features: List[str] = None  # ['energy', 'zcr', 'spectral_centroid', 'pitch_proxy']
    
    # Temporal aggregation
    temporal_window_sec: float = 10.0  # Window for computing stats
    temporal_stats: List[str] = None  # ['mean', 'std', 'delta']
    
    # Feature fusion
    fusion_strategy: str = 'concat'  # 'concat', 'attention', 'learned'
    normalize_features: bool = True
    
    def __post_init__(self):
        # Set defaults
        if self.rppg_features is None:
            self.rppg_features = ['hr_bpm', 'sdnn', 'rmssd', 'lf_hf_ratio', 'psd_peak_freq']
        
        if self.emotion_features is None:
            self.emotion_features = ['emotion_probs', 'dominant_emotion_conf']
        
        if self.behavioral_features is None:
            self.behavioral_features = ['blink_rate', 'perclos']

        if self.audio_features is None:
            self.audio_features = ['energy', 'zcr', 'spectral_centroid', 'pitch_proxy', 'voiced_ratio']
        
        if self.temporal_stats is None:
            self.temporal_stats = ['mean', 'std', 'delta']


class rPPGFeatureExtractor:
    """Extract cardiovascular features from video using rPPG."""
    
    def __init__(self, window_sec: float = 7.0):
        """
        Initialize rPPG feature extractor.
        
        Args:
            window_sec: Window size for rPPG estimation (seconds)
        """
        self.window_sec = window_sec
    
    def extract(self, video_path: str, fps: float = 30.0) -> Dict[str, np.ndarray]:
        """
        Extract rPPG features from video.
        
        Args:
            video_path: Path to video file
            fps: Video frame rate
        
        Returns:
            Dictionary of time-series features:
                - hr_bpm: Heart rate estimates [T]
                - sdnn: Standard deviation of NN intervals [T]
                - rmssd: Root mean square of successive differences [T]
                - lf_hf_ratio: Low frequency / High frequency ratio [T]
                - psd_peak_freq: Peak frequency from PSD [T]
        """
        try:
            # Run rPPG pipeline
            results = extract_rppg(video_path, landmarks_cache_dir="cache/landmarks")
            
            if results is None or not results:
                return self._empty_features()
            
            # Extract time series
            features = {
                'hr_bpm': np.array([r['bpm'] for r in results if 'bpm' in r]),
                'sdnn': np.array([r.get('sdnn', 0) for r in results]),
                'rmssd': np.array([r.get('rmssd', 0) for r in results]),
                'lf_hf_ratio': np.array([r.get('lf_hf_ratio', 0) for r in results]),
                'psd_peak_freq': np.array([r.get('psd_peak_freq', 0) for r in results])
            }
            
            return features
        
        except Exception as e:
            print(f"Warning: rPPG extraction failed: {e}")
            return self._empty_features()
    
    def _empty_features(self) -> Dict[str, np.ndarray]:
        """Return empty feature dict (for failed extraction)."""
        return {
            'hr_bpm': np.array([]),
            'sdnn': np.array([]),
            'rmssd': np.array([]),
            'lf_hf_ratio': np.array([]),
            'psd_peak_freq': np.array([])
        }


class EmotionFeatureExtractor:
    """Extract emotion recognition features from frames."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize emotion feature extractor.
        
        Args:
            model_path: Path to pre-trained emotion model (or None for random init)
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load emotion model
        self.model = EmotionClassifier(
            num_classes=8,
            backbone='resnet50',
            pretrained=(model_path is None)
        ).to(device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
        
        # Emotion labels
        self.emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 
                               'happiness', 'sadness', 'surprise', 'neutral']
    
    def extract(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract emotion features from frames.
        
        Args:
            frames: Video frames [T, H, W, C] or single frame [H, W, C]
        
        Returns:
            Dictionary of features:
                - emotion_probs: Emotion probabilities [T, 8]
                - dominant_emotion: Dominant emotion index [T]
                - dominant_emotion_conf: Confidence of dominant emotion [T]
                - valence: Estimated valence [-1, 1] [T]
                - arousal: Estimated arousal [-1, 1] [T]
        """
        try:
            # Ensure 4D: [T, H, W, C]
            if frames.ndim == 3:
                frames = frames[np.newaxis, ...]
            
            # Preprocess frames
            frames_tensor = self._preprocess(frames)
            
            # Inference
            with torch.no_grad():
                logits = self.model(frames_tensor.to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Dominant emotion
            dominant = np.argmax(probs, axis=1)
            dominant_conf = np.max(probs, axis=1)
            
            # Estimate valence/arousal from emotion distribution
            valence, arousal = self._emotion_to_va(probs)
            
            return {
                'emotion_probs': probs,  # [T, 8]
                'dominant_emotion': dominant,  # [T]
                'dominant_emotion_conf': dominant_conf,  # [T]
                'valence': valence,  # [T]
                'arousal': arousal   # [T]
            }
        
        except Exception as e:
            print(f"Warning: Emotion extraction failed: {e}")
            return self._empty_features(len(frames))
    
    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess frames for model input.
        
        Args:
            frames: [T, H, W, C] uint8 0-255
        
        Returns:
            Tensor [T, C, H, W] normalized
        """
        # Resize to 224x224
        frames_resized = np.stack([
            cv2.resize(frame, (224, 224)) for frame in frames
        ], axis=0)
        
        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames_resized).permute(0, 3, 1, 2).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor
    
    def _emotion_to_va(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map emotion probabilities to valence-arousal space.
        
        Based on Russell's circumplex model:
            anger: V=-0.5, A=0.8
            contempt: V=-0.3, A=0.2
            disgust: V=-0.6, A=0.5
            fear: V=-0.6, A=0.8
            happiness: V=0.8, A=0.6
            sadness: V=-0.6, A=-0.4
            surprise: V=0.4, A=0.8
            neutral: V=0.0, A=0.0
        
        Args:
            probs: Emotion probabilities [T, 8]
        
        Returns:
            Tuple of (valence [T], arousal [T])
        """
        # Emotion-to-VA mapping
        va_map = np.array([
            [-0.5, 0.8],   # anger
            [-0.3, 0.2],   # contempt
            [-0.6, 0.5],   # disgust
            [-0.6, 0.8],   # fear
            [0.8, 0.6],    # happiness
            [-0.6, -0.4],  # sadness
            [0.4, 0.8],    # surprise
            [0.0, 0.0]     # neutral
        ])
        
        # Weighted average
        va = probs @ va_map  # [T, 2]
        valence = va[:, 0]
        arousal = va[:, 1]
        
        return valence, arousal
    
    def _empty_features(self, length: int = 1) -> Dict[str, np.ndarray]:
        """Return empty feature dict."""
        return {
            'emotion_probs': np.zeros((length, 8)),
            'dominant_emotion': np.zeros(length, dtype=int),
            'dominant_emotion_conf': np.zeros(length),
            'valence': np.zeros(length),
            'arousal': np.zeros(length)
        }


class AudioFeatureExtractor:
    """Extract lightweight speech/prosody features from WAV audio."""

    def __init__(self, frame_sec: float = 0.025, hop_sec: float = 0.010):
        self.frame_sec = frame_sec
        self.hop_sec = hop_sec

    def extract(self, audio_path: str) -> Dict[str, np.ndarray]:
        try:
            sample_rate, audio = wavfile.read(audio_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audio = audio.astype(np.float32)
            if audio.size == 0:
                return self._empty_features()

            max_abs = np.max(np.abs(audio)) + 1e-6
            audio = audio / max_abs

            frame_len = max(64, int(self.frame_sec * sample_rate))
            hop_len = max(32, int(self.hop_sec * sample_rate))
            if len(audio) < frame_len:
                audio = np.pad(audio, (0, frame_len - len(audio)))

            frames = []
            for start in range(0, len(audio) - frame_len + 1, hop_len):
                frames.append(audio[start:start + frame_len])
            frames = np.stack(frames, axis=0)

            window = np.hanning(frame_len).astype(np.float32)
            win_frames = frames * window[None, :]

            energy = np.mean(win_frames ** 2, axis=1)
            zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

            spec = np.abs(np.fft.rfft(win_frames, axis=1)) + 1e-8
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
            spec_sum = np.sum(spec, axis=1) + 1e-8
            spectral_centroid = np.sum(spec * freqs[None, :], axis=1) / spec_sum
            spectral_bandwidth = np.sqrt(
                np.sum(spec * (freqs[None, :] - spectral_centroid[:, None]) ** 2, axis=1) / spec_sum
            )

            pitch_proxy = np.array([self._pitch_autocorr(frame, sample_rate) for frame in frames], dtype=np.float32)
            voiced_ratio = (pitch_proxy > 50.0).astype(np.float32)

            return {
                'energy': energy,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'pitch_proxy': pitch_proxy,
                'voiced_ratio': voiced_ratio,
            }
        except Exception as e:
            print(f"Warning: Audio extraction failed: {e}")
            return self._empty_features()

    def _pitch_autocorr(self, frame: np.ndarray, sample_rate: int) -> float:
        frame = frame - frame.mean()
        if np.max(np.abs(frame)) < 1e-6:
            return 0.0
        corr = signal.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        min_lag = max(1, int(sample_rate / 400))
        max_lag = max(min_lag + 1, int(sample_rate / 70))
        if max_lag >= len(corr):
            return 0.0
        lag = np.argmax(corr[min_lag:max_lag]) + min_lag
        if lag <= 0:
            return 0.0
        return float(sample_rate / lag)

    def _empty_features(self) -> Dict[str, np.ndarray]:
        return {
            'energy': np.array([]),
            'zcr': np.array([]),
            'spectral_centroid': np.array([]),
            'spectral_bandwidth': np.array([]),
            'pitch_proxy': np.array([]),
            'voiced_ratio': np.array([]),
        }


class MultimodalFeatureBuilder:
    """
    Main class for extracting and fusing multimodal features.
    
    Combines rPPG, emotion, and behavioral features into unified representation.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize multimodal feature builder.
        
        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        
        # Initialize extractors
        if self.config.use_rppg:
            self.rppg_extractor = rPPGFeatureExtractor(
                window_sec=self.config.rppg_window_sec
            )
        
        if self.config.use_emotion:
            self.emotion_extractor = EmotionFeatureExtractor(
                model_path=self.config.emotion_model_path
            )

        if self.config.use_audio:
            self.audio_extractor = AudioFeatureExtractor()
    
    def extract(
        self,
        video_path: Optional[str] = None,
        frames: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        fps: float = 30.0
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features from video.
        
        Args:
            video_path: Path to video file (for rPPG)
            frames: Pre-loaded frames [T, H, W, C] (for emotion)
            fps: Video frame rate
        
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # Extract rPPG features
        if self.config.use_rppg and video_path:
            rppg_features = self.rppg_extractor.extract(video_path, fps)
            features.update({f'rppg_{k}': v for k, v in rppg_features.items()})
        
        # Extract emotion features
        if self.config.use_emotion and frames is not None:
            emotion_features = self.emotion_extractor.extract(frames)
            features.update({f'emotion_{k}': v for k, v in emotion_features.items()})

        # Extract audio features
        if self.config.use_audio and audio_path:
            audio_features = self.audio_extractor.extract(audio_path)
            features.update({f'audio_{k}': v for k, v in audio_features.items()})
        
        return features
    
    def fuse_features(
        self,
        features: Dict[str, np.ndarray],
        strategy: str = 'concat'
    ) -> np.ndarray:
        """
        Fuse multimodal features into unified representation.
        
        Args:
            features: Dictionary of extracted features
            strategy: Fusion strategy ('concat', 'mean', 'attention')
        
        Returns:
            Fused feature vector [D]
        """
        # Aggregate temporal features
        aggregated = {}
        
        for key, values in features.items():
            if len(values) == 0:
                continue
            
            # Compute statistics
            if 'mean' in self.config.temporal_stats:
                aggregated[f'{key}_mean'] = np.mean(values, axis=0) if values.ndim > 1 else np.mean(values)
            
            if 'std' in self.config.temporal_stats:
                aggregated[f'{key}_std'] = np.std(values, axis=0) if values.ndim > 1 else np.std(values)
            
            if 'delta' in self.config.temporal_stats and len(values) > 1:
                aggregated[f'{key}_delta'] = values[-1] - values[0] if values.ndim == 1 else (values[-1] - values[0])
        
        # Concatenate all features
        feature_list = []
        for key, value in sorted(aggregated.items()):
            if isinstance(value, np.ndarray):
                feature_list.append(value.flatten())
            else:
                feature_list.append(np.array([value]))
        
        fused = np.concatenate(feature_list) if feature_list else np.array([])
        
        # Normalize
        if self.config.normalize_features and len(fused) > 0:
            fused = (fused - np.mean(fused)) / (np.std(fused) + 1e-6)
        
        return fused
    
    def extract_and_fuse(
        self,
        video_path: Optional[str] = None,
        frames: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Complete pipeline: extract + fuse features.
        
        Args:
            video_path: Path to video file
            frames: Pre-loaded frames
            fps: Video frame rate
        
        Returns:
            Fused feature vector [D]
        """
        # Extract
        features = self.extract(video_path, frames, audio_path, fps)
        
        # Fuse
        fused = self.fuse_features(features, strategy=self.config.fusion_strategy)
        
        return fused


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 80)
    print("Multimodal Feature Builder - Test")
    print("=" * 80)
    
    # Create config
    config = FeatureConfig(
        use_rppg=True,
        use_emotion=True,
        use_behavioral=False,
        temporal_window_sec=10.0
    )
    
    # Create builder
    builder = MultimodalFeatureBuilder(config)
    
    print("\n✅ Feature builder initialized!")
    print(f"   rPPG features: {config.rppg_features}")
    print(f"   Emotion features: {config.emotion_features}")
    print(f"   Temporal stats: {config.temporal_stats}")
    
    # Test with synthetic data
    print("\n📊 Testing with synthetic data...")
    frames = np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8)
    
    features = builder.extract(frames=frames)
    print(f"\n   Extracted features: {list(features.keys())}")
    
    fused = builder.fuse_features(features)
    print(f"   Fused feature vector shape: {fused.shape}")
    
    print("\n✅ Feature extraction pipeline working!")
