"""
SAMM Micro-Expression Dataset Loader
=====================================
Professional-grade PyTorch dataset for SAMM (Spontaneous Actions and Micro-Movements)
micro-expression database with temporal sequence handling and FACS code integration.

Dataset Structure:
    data/SAMM/
        {subject_id}/
            {subject_id}_{video_id}/
                {subject_id}_{frame_num}.jpg
        SAMM_Micro_FACS_Codes_v2.xlsx  # Annotations

Features:
    • Temporal sequence loading with onset/apex/offset frames
    • FACS Action Unit (AU) code parsing
    • Emotion label mapping from AU patterns
    • Frame-level and sequence-level data access
    • rPPG-compatible preprocessing pipeline
    • Train/val/test split support with subject-based stratification
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import warnings


class SAMMConfig:
    """Configuration for SAMM dataset processing."""
    
    # Dataset paths
    ROOT_DIR = Path("data/SAMM")
    ANNOTATION_FILE = ROOT_DIR / "SAMM_Micro_FACS_Codes_v2.xlsx"
    
    # Emotion mapping from FACS Action Units
    # Based on EMFACS (Emotional Facial Action Coding System)
    AU_TO_EMOTION = {
        # Happiness: AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)
        ('6', '12'): 'happiness',
        ('12',): 'happiness',
        
        # Sadness: AU1 (Inner Brow Raiser) + AU4 (Brow Lowerer) + AU15 (Lip Corner Depressor)
        ('1', '4', '15'): 'sadness',
        ('4', '15'): 'sadness',
        
        # Surprise: AU1 + AU2 (Outer Brow Raiser) + AU5 (Upper Lid Raiser) + AU26 (Jaw Drop)
        ('1', '2', '5'): 'surprise',
        ('1', '2', '26'): 'surprise',
        ('5', '26'): 'surprise',
        
        # Fear: AU1 + AU2 + AU4 + AU5 + AU20 (Lip Stretcher) + AU26
        ('1', '2', '4', '5'): 'fear',
        ('1', '4', '5', '20'): 'fear',
        
        # Anger: AU4 + AU5 + AU7 (Lid Tightener) + AU23 (Lip Tightener)
        ('4', '7'): 'anger',
        ('4', '5', '7'): 'anger',
        ('4', '7', '23'): 'anger',
        
        # Disgust: AU9 (Nose Wrinkler) + AU15 + AU16 (Lower Lip Depressor)
        ('9',): 'disgust',
        ('9', '15'): 'disgust',
        ('9', '16'): 'disgust',
        
        # Contempt: AU12 + AU14 (Dimpler) - unilateral
        ('14',): 'contempt',
        ('12', '14'): 'contempt',
    }
    
    # Standard emotion categories (aligned with CASMEII)
    EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
    
    # Frame selection strategy
    SEQUENCE_LENGTH = 16  # Number of frames to sample per micro-expression
    FRAME_SIZE = (224, 224)  # Standard input size for emotion recognition models
    
    # Temporal sampling
    SAMPLE_STRATEGY = 'uniform'  # 'uniform', 'onset_apex_offset', 'dense'


class SAMMAnnotationParser:
    """Parser for SAMM annotation Excel file."""
    
    def __init__(self, annotation_path: Path):
        """
        Initialize annotation parser.
        
        Args:
            annotation_path: Path to SAMM_Micro_FACS_Codes_v2.xlsx
        """
        self.annotation_path = annotation_path
        self._annotations = None
        self._load_annotations()
    
    def _load_annotations(self):
        """Load and parse Excel annotations."""
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"SAMM annotation file not found: {self.annotation_path}\n"
                f"Please ensure SAMM_Micro_FACS_Codes_v2.xlsx is in {self.annotation_path.parent}"
            )
        
        try:
            # Read Excel file with skiprows=12 (SAMM format has headers at row 13)
            # First row after skip is the header row
            df = pd.read_excel(self.annotation_path, engine='openpyxl', skiprows=12, header=0)
            
            # First row contains headers, use it
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            
            # Standardize column names
            df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
            
            # Rename columns to expected format
            column_mapping = {
                'onset_frame': 'onset',
                'apex_frame': 'apex',
                'offset_frame': 'offset'
            }
            df = df.rename(columns=column_mapping)
            
            # Remove rows with NaN subject (extra rows at end)
            df = df[df['subject'].notna()]
            
            # Convert frame numbers to integers
            for col in ['onset', 'apex', 'offset']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Expected columns: subject, filename, onset, apex, offset
            required_cols = ['subject', 'filename', 'onset', 'apex', 'offset']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                print(f"Available columns: {df.columns.tolist()}")
            
            self._annotations = df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SAMM annotations: {e}")
    
    def get_annotations(self) -> pd.DataFrame:
        """Get full annotation dataframe."""
        return self._annotations.copy()
    
    def get_sequence_info(self, subject_id: str, filename: str) -> Optional[Dict]:
        """
        Get annotation info for a specific micro-expression sequence.
        
        Args:
            subject_id: Subject ID (e.g., '006')
            filename: Video sequence name (e.g., '006_1_2')
        
        Returns:
            Dictionary with onset, apex, offset frames and AUs, or None if not found
        """
        # Try different column name variations
        subject_col = next((c for c in self._annotations.columns if 'subject' in c), None)
        filename_col = next((c for c in self._annotations.columns if 'filename' in c or 'induced' in c), None)
        
        if subject_col is None or filename_col is None:
            warnings.warn("Cannot find subject/filename columns in annotations")
            return None
        
        # Query annotation
        mask = (self._annotations[subject_col].astype(str).str.zfill(3) == str(subject_id)) & \
               (self._annotations[filename_col].astype(str) == filename)
        
        rows = self._annotations[mask]
        
        if len(rows) == 0:
            return None
        
        row = rows.iloc[0]
        
        # Extract frame indices
        onset = int(row.get('onset', 0)) if pd.notna(row.get('onset')) else 0
        apex = int(row.get('apex', onset)) if pd.notna(row.get('apex')) else onset
        offset = int(row.get('offset', apex)) if pd.notna(row.get('offset')) else apex
        
        # Extract Action Units
        au_col = next((c for c in row.index if 'action' in c or 'au' in c or 'facs' in c), None)
        action_units = str(row.get(au_col, '')).strip() if au_col and pd.notna(row.get(au_col)) else ''
        
        # Extract emotion label if available
        emotion_col = next((c for c in row.index if 'emotion' in c or 'estimated' in c), None)
        emotion = str(row.get(emotion_col, '')).strip().lower() if emotion_col and pd.notna(row.get(emotion_col)) else 'neutral'
        
        return {
            'onset': onset,
            'apex': apex,
            'offset': offset,
            'action_units': action_units,
            'emotion': emotion,
            'duration': offset - onset + 1
        }
    
    def map_au_to_emotion(self, action_units: str) -> str:
        """
        Map FACS Action Units to emotion category.
        
        Args:
            action_units: String of AU codes (e.g., "4+7" or "12")
        
        Returns:
            Emotion label (from SAMMConfig.EMOTION_LABELS)
        """
        if not action_units or pd.isna(action_units):
            return 'neutral'
        
        # Parse AU codes: handle formats like "4+7", "R12", "L14", "1+2+5"
        aus = str(action_units).replace('R', '').replace('L', '').replace('+', ' ').replace(',', ' ').split()
        aus = [au.strip() for au in aus if au.strip().isdigit()]
        aus_tuple = tuple(sorted(set(aus)))
        
        # Check exact matches first
        if aus_tuple in SAMMConfig.AU_TO_EMOTION:
            return SAMMConfig.AU_TO_EMOTION[aus_tuple]
        
        # Check subsets (if AU pattern contains known emotion pattern)
        for au_pattern, emotion in SAMMConfig.AU_TO_EMOTION.items():
            if set(au_pattern).issubset(set(aus_tuple)):
                return emotion
        
        # Default to neutral if no match
        return 'neutral'


class SAMMDataset(Dataset):
    """
    PyTorch Dataset for SAMM micro-expression sequences.
    
    Supports:
        • Temporal sequence loading with configurable sampling
        • Frame-level and sequence-level data access
        • Train/val/test splits with subject-based stratification
        • Data augmentation for training
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path] = SAMMConfig.ROOT_DIR,
        split: str = 'train',
        sequence_length: int = SAMMConfig.SEQUENCE_LENGTH,
        frame_size: Tuple[int, int] = SAMMConfig.FRAME_SIZE,
        sample_strategy: str = SAMMConfig.SAMPLE_STRATEGY,
        transform=None,
        use_rppg: bool = True,
        subjects: Optional[List[str]] = None,
        enable_augmentation: bool = True,
        augmentation_strength: float = 1.0
    ):
        """
        Initialize SAMM dataset.
        
        Args:
            root_dir: Root directory containing SAMM data
            split: Dataset split ('train', 'val', 'test')
            sequence_length: Number of frames to sample per sequence
            frame_size: Target frame size (H, W)
            sample_strategy: Frame sampling strategy
            transform: Optional transforms (albumentations or torchvision)
            use_rppg: Whether to enable rPPG-compatible preprocessing
            subjects: Optional list of subject IDs to include (overrides split)
            enable_augmentation: Enable training-time augmentation for split='train'
            augmentation_strength: Global augmentation strength multiplier (0.0-2.0)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.sample_strategy = sample_strategy
        self.transform = transform
        self.use_rppg = use_rppg
        self.enable_augmentation = enable_augmentation
        self.augmentation_strength = max(0.0, min(2.0, float(augmentation_strength)))
        
        # Load annotations
        self.parser = SAMMAnnotationParser(SAMMConfig.ANNOTATION_FILE)
        
        # Build dataset index
        self.samples = self._build_dataset_index(subjects)
        
        if len(self.samples) == 0:
            warnings.warn(f"No valid samples found for split '{split}'")
    
    def _build_dataset_index(self, subjects: Optional[List[str]] = None) -> List[Dict]:
        """
        Build index of all valid micro-expression sequences.
        
        Args:
            subjects: Optional list of subject IDs to include
        
        Returns:
            List of sample dictionaries with paths and metadata
        """
        samples = []
        
        # Get all subject directories
        subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        
        # Apply subject filter or split
        if subjects is not None:
            subject_dirs = [d for d in subject_dirs if d.name in subjects]
        else:
            # Default train/val/test split (roughly 70/15/15)
            all_subjects = [d.name for d in subject_dirs]
            n = len(all_subjects)
            
            if self.split == 'train':
                selected = all_subjects[:int(0.7 * n)]
            elif self.split == 'val':
                selected = all_subjects[int(0.7 * n):int(0.85 * n)]
            else:  # test
                selected = all_subjects[int(0.85 * n):]
            
            subject_dirs = [d for d in subject_dirs if d.name in selected]
        
        # Index all sequences
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            
            # Get all video sequences for this subject
            video_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                video_name = video_dir.name
                
                # Get frames
                frames = sorted(video_dir.glob(f"{subject_id}_*.jpg"))
                
                if len(frames) == 0:
                    continue
                
                # Get annotation info
                anno_info = self.parser.get_sequence_info(subject_id, video_name)
                
                if anno_info is None:
                    # No annotation found, use all frames
                    onset, apex, offset = 0, len(frames) // 2, len(frames) - 1
                    emotion = 'neutral'
                    action_units = ''
                else:
                    onset = anno_info['onset']
                    apex = anno_info['apex']
                    offset = anno_info['offset']
                    action_units = anno_info['action_units']
                    
                    # Map AU to emotion
                    emotion = self.parser.map_au_to_emotion(action_units)
                    
                    # Use annotated emotion if available
                    if anno_info['emotion'] and anno_info['emotion'] != 'neutral':
                        emotion = anno_info['emotion']
                
                samples.append({
                    'subject_id': subject_id,
                    'video_name': video_name,
                    'video_path': video_dir,
                    'frames': frames,
                    'onset': onset,
                    'apex': apex,
                    'offset': offset,
                    'emotion': emotion,
                    'action_units': action_units,
                    'emotion_idx': SAMMConfig.EMOTION_TO_IDX.get(emotion, SAMMConfig.EMOTION_TO_IDX['neutral'])
                })
        
        return samples
    
    def _sample_frames(self, sample: Dict) -> List[Path]:
        """
        Sample frames from sequence based on strategy.
        
        Args:
            sample: Sample dictionary from dataset index
        
        Returns:
            List of frame paths to load
        """
        frames = sample['frames']
        onset = sample['onset']
        apex = sample['apex']
        offset = sample['offset']
        
        # Clamp indices to valid range
        onset = max(0, min(onset, len(frames) - 1))
        apex = max(onset, min(apex, len(frames) - 1))
        offset = max(apex, min(offset, len(frames) - 1))
        
        if self.sample_strategy == 'onset_apex_offset':
            # Sample key frames: onset, apex, offset + interpolations
            key_frames = [onset, apex, offset]
            
            # Fill remaining with uniform sampling
            if self.sequence_length > 3:
                # Sample uniformly between key frames
                between_onset_apex = np.linspace(onset, apex, self.sequence_length // 2, dtype=int)
                between_apex_offset = np.linspace(apex, offset, self.sequence_length - self.sequence_length // 2, dtype=int)
                indices = sorted(set(list(between_onset_apex) + list(between_apex_offset)))[:self.sequence_length]
            else:
                indices = key_frames[:self.sequence_length]
        
        elif self.sample_strategy == 'dense':
            # Sample densely around apex
            start_idx = max(0, apex - self.sequence_length // 2)
            end_idx = min(len(frames), start_idx + self.sequence_length)
            indices = list(range(start_idx, end_idx))
        
        else:  # uniform (default)
            # Uniform sampling from onset to offset
            indices = np.linspace(onset, offset, self.sequence_length, dtype=int)
        
        # Ensure we have exactly sequence_length frames
        while len(indices) < self.sequence_length:
            indices.append(indices[-1])  # Repeat last frame
        
        indices = indices[:self.sequence_length]
        
        return [frames[idx] for idx in indices]

    def _apply_temporal_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """Apply light temporal jitter while preserving sequence length."""
        if frames.shape[0] <= 1:
            return frames

        p_jitter = 0.35 * self.augmentation_strength
        if np.random.random() < p_jitter:
            t = frames.shape[0]
            base_idx = np.linspace(0, t - 1, t)
            jitter = np.random.uniform(-1.0, 1.0, size=t) * self.augmentation_strength
            indices = np.clip(np.round(base_idx + jitter), 0, t - 1).astype(int)
            frames = frames[indices]

        return frames

    def _apply_spatial_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporally consistent spatial augmentation for training."""
        out = frames

        # Random horizontal flip.
        if np.random.random() < 0.5:
            out = out[:, :, ::-1, :].copy()

        # Brightness and contrast are applied with shared factors for all frames.
        if np.random.random() < 0.7:
            brightness = 1.0 + np.random.uniform(-0.15, 0.15) * self.augmentation_strength
            out = np.clip(out.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if np.random.random() < 0.7:
            contrast = 1.0 + np.random.uniform(-0.15, 0.15) * self.augmentation_strength
            mean_val = out.astype(np.float32).mean()
            out = np.clip((out.astype(np.float32) - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        # Small in-plane rotation (same angle for temporal consistency).
        if np.random.random() < 0.4:
            angle = np.random.uniform(-6.0, 6.0) * self.augmentation_strength
            if abs(angle) > 0.25:
                h, w = out.shape[1], out.shape[2]
                mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                out = np.stack([
                    cv2.warpAffine(f, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    for f in out
                ])

        # Mild gaussian noise.
        if np.random.random() < 0.25:
            noise_std = np.random.uniform(2.0, 8.0) * self.augmentation_strength
            noise = np.random.normal(0.0, noise_std, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Mild blur.
        if np.random.random() < 0.2:
            out = np.stack([cv2.GaussianBlur(f, (3, 3), 0) for f in out])

        return out
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
                - frames: Tensor of shape [T, C, H, W] (temporal, channels, height, width)
                - emotion_label: Integer emotion class
                - emotion_name: String emotion name
                - subject_id: Subject identifier
                - video_name: Video sequence name
                - frame_paths: List of loaded frame paths (for rPPG processing)
        """
        sample = self.samples[idx]
        
        # Sample frames
        frame_paths = self._sample_frames(sample)
        
        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            if img is None:
                # Fallback: use blank frame
                img = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.frame_size)
            
            frames.append(img)
        
        frames = np.stack(frames, axis=0)  # Shape: [T, H, W, C]
        
        # Apply transforms
        if self.transform:
            # Apply same transform to all frames
            frames = np.stack([self.transform(image=f)['image'] for f in frames], axis=0)
        
        # Training augmentation (temporal + spatial, sequence-consistent)
        if self.split == 'train' and self.enable_augmentation:
            frames = self._apply_temporal_augmentation(frames)
            frames = self._apply_spatial_augmentation(frames)
        
        # Convert to tensor: [T, H, W, C] -> [T, C, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        return {
            'frames': frames,
            'emotion_label': torch.tensor(sample['emotion_idx'], dtype=torch.long),
            'emotion_name': sample['emotion'],
            'subject_id': sample['subject_id'],
            'video_name': sample['video_name'],
            'frame_paths': [str(p) for p in frame_paths]  # For rPPG processing
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Returns:
            Tensor of weights for each emotion class
        """
        emotion_counts = np.bincount([s['emotion_idx'] for s in self.samples], minlength=len(SAMMConfig.EMOTION_LABELS))
        total = emotion_counts.sum()
        weights = total / (len(SAMMConfig.EMOTION_LABELS) * emotion_counts + 1e-6)  # Avoid division by zero
        return torch.from_numpy(weights).float()


def get_samm_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    sequence_length: int = 16,
    enable_augmentation: bool = True,
    augmentation_strength: float = 1.0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders for SAMM.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        sequence_length: Frames per sequence
        enable_augmentation: Enable train-time augmentation
        augmentation_strength: Train-time augmentation strength
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SAMMDataset(
        split='train',
        sequence_length=sequence_length,
        enable_augmentation=enable_augmentation,
        augmentation_strength=augmentation_strength
    )
    val_dataset = SAMMDataset(
        split='val',
        sequence_length=sequence_length,
        enable_augmentation=False
    )
    test_dataset = SAMMDataset(
        split='test',
        sequence_length=sequence_length,
        enable_augmentation=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("=" * 80)
    print("SAMM Dataset Loader - Validation Test")
    print("=" * 80)
    
    try:
        # Create dataset
        dataset = SAMMDataset(split='train', sequence_length=8)
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            
            print(f"\n📊 Sample structure:")
            print(f"   Frames shape: {sample['frames'].shape}")
            print(f"   Emotion: {sample['emotion_name']} (idx={sample['emotion_label'].item()})")
            print(f"   Subject: {sample['subject_id']}")
            print(f"   Video: {sample['video_name']}")
            
            # Class distribution
            from collections import Counter
            emotions = [s['emotion'] for s in dataset.samples]
            emotion_counts = Counter(emotions)
            
            print(f"\n📈 Emotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                print(f"   {emotion:12s}: {count:3d} samples ({count/len(dataset)*100:.1f}%)")
            
            # Class weights
            weights = dataset.get_class_weights()
            print(f"\n⚖️ Class weights (for imbalanced training):")
            for i, (emotion, weight) in enumerate(zip(SAMMConfig.EMOTION_LABELS, weights)):
                print(f"   {emotion:12s}: {weight:.3f}")
        
        print(f"\n✅ All checks passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
