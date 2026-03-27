"""
CASMEII Emotion Dataset Loader
===============================
Professional-grade PyTorch dataset for CASMEII (Chinese Academy of Sciences 
Macro-Expression Database II) with emotion classification support.

Dataset Structure:
    data/CASMEII/
        train/
            {emotion}/
                {image_id}.jpg
        test/
            {emotion}/
                {image_id}.jpg

Features:
    • Standard emotion classification (5+ classes)
    • Pre-split train/test organization
    • Augmentation pipeline for training
    • Integration with SAMM emotion labels
    • Weighted sampling for class balance
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import Counter
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import warnings


class CASMEIIConfig:
    """Configuration for CASMEII dataset processing."""
    
    # Dataset paths
    ROOT_DIR = Path("data/CASMEII")
    TRAIN_DIR = ROOT_DIR / "train"
    TEST_DIR = ROOT_DIR / "test"
    
    # Emotion categories (standardized with SAMM)
    # CASMEII may have: angry, disgust, fear, happy, neutral, sad, surprise
    EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Label mapping (handle variations in folder names)
    LABEL_MAPPING = {
        'angry': 'anger',
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happiness',
        'happiness': 'happiness',
        'sad': 'sadness',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral',
        'contempt': 'contempt',
        'repression': 'neutral',  # Map rare classes to neutral
        'others': 'neutral'
    }
    
    EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
    
    # Image preprocessing
    FRAME_SIZE = (224, 224)  # Standard input size
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)  # ImageNet stats
    NORMALIZE_STD = (0.229, 0.224, 0.225)


class CASMEIIDataset(Dataset):
    """
    PyTorch Dataset for CASMEII emotion classification.
    
    Supports:
        • Static image emotion classification
        • Train/test splits
        • Data augmentation for training
        • Class balancing
        • Compatible with SAMM emotion labels
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path] = CASMEIIConfig.ROOT_DIR,
        split: str = 'train',
        frame_size: Tuple[int, int] = CASMEIIConfig.FRAME_SIZE,
        transform=None,
        balance_classes: bool = False
    ):
        """
        Initialize CASMEII dataset.
        
        Args:
            root_dir: Root directory containing CASMEII data
            split: Dataset split ('train' or 'test')
            frame_size: Target frame size (H, W)
            transform: Optional transforms (albumentations or torchvision)
            balance_classes: Whether to balance classes during sampling
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.frame_size = frame_size
        self.transform = transform
        self.balance_classes = balance_classes
        
        # Set data directory
        if split == 'train':
            self.data_dir = self.root_dir / 'train'
        else:
            self.data_dir = self.root_dir / 'test'
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"CASMEII {split} directory not found: {self.data_dir}\n"
                f"Please ensure data is organized as: data/CASMEII/train/ and data/CASMEII/test/"
            )
        
        # Build dataset index
        self.samples = self._build_dataset_index()
        
        if len(self.samples) == 0:
            warnings.warn(f"No valid samples found in {self.data_dir}")
    
    def _build_dataset_index(self) -> List[Dict]:
        """
        Build index of all emotion images.
        
        Returns:
            List of sample dictionaries with paths and labels
        """
        samples = []
        
        # Get emotion directories
        emotion_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for emotion_dir in emotion_dirs:
            emotion_raw = emotion_dir.name.lower()
            
            # Map to standard emotion label
            emotion = CASMEIIConfig.LABEL_MAPPING.get(emotion_raw, 'neutral')
            emotion_idx = CASMEIIConfig.EMOTION_TO_IDX.get(emotion, CASMEIIConfig.EMOTION_TO_IDX['neutral'])
            
            # Get all images in this emotion category
            image_files = list(emotion_dir.glob('*.jpg')) + \
                         list(emotion_dir.glob('*.jpeg')) + \
                         list(emotion_dir.glob('*.png'))
            
            for image_path in image_files:
                samples.append({
                    'image_path': image_path,
                    'emotion': emotion,
                    'emotion_idx': emotion_idx,
                    'emotion_raw': emotion_raw,
                    'image_id': image_path.stem
                })
        
        return samples
    
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
                - image: Tensor of shape [C, H, W]
                - emotion_label: Integer emotion class
                - emotion_name: String emotion name
                - image_id: Image identifier
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        img = cv2.imread(str(image_path))
        
        if img is None:
            # Fallback: blank image
            img = np.zeros((*self.frame_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.frame_size)
        
        # Apply transforms
        if self.transform:
            img = self.transform(image=img)['image']
        
        # Convert to tensor: [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor(CASMEIIConfig.NORMALIZE_MEAN).view(3, 1, 1)
        std = torch.tensor(CASMEIIConfig.NORMALIZE_STD).view(3, 1, 1)
        img = (img - mean) / std
        
        return {
            'image': img,
            'emotion_label': torch.tensor(sample['emotion_idx'], dtype=torch.long),
            'emotion_name': sample['emotion'],
            'image_id': sample['image_id']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Returns:
            Tensor of weights for each emotion class
        """
        emotion_counts = np.bincount(
            [s['emotion_idx'] for s in self.samples],
            minlength=len(CASMEIIConfig.EMOTION_LABELS)
        )
        # Handle empty classes by using at least 1 sample
        emotion_counts = np.maximum(emotion_counts, 1)
        total = emotion_counts.sum()
        weights = total / (len(CASMEIIConfig.EMOTION_LABELS) * emotion_counts)
        return torch.from_numpy(weights).float()
    
    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """
        Create weighted sampler for balanced training.
        
        Returns:
            WeightedRandomSampler or None
        """
        if not self.balance_classes or self.split != 'train':
            return None
        
        # Compute sample weights
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[s['emotion_idx']].item() for s in self.samples]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get emotion class distribution.
        
        Returns:
            Dictionary mapping emotion names to counts
        """
        emotions = [s['emotion'] for s in self.samples]
        return dict(Counter(emotions))


def get_casmeii_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    balance_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for CASMEII.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        balance_train: Whether to balance training classes
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = CASMEIIDataset(split='train', balance_classes=balance_train)
    test_dataset = CASMEIIDataset(split='test', balance_classes=False)
    
    # Get sampler for balanced training
    train_sampler = train_dataset.get_sampler() if balance_train else None
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
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
    
    return train_loader, test_loader


class UnifiedEmotionDataset(Dataset):
    """
    Unified dataset combining SAMM and CASMEII for joint training.
    
    Features:
        • Combines temporal (SAMM) and static (CASMEII) data
        • Unified emotion label space
        • Handles different data modalities
    """
    
    def __init__(
        self,
        samm_dataset: Optional[Dataset] = None,
        casmeii_dataset: Optional[Dataset] = None,
        sample_ratio: float = 0.5
    ):
        """
        Initialize unified dataset.
        
        Args:
            samm_dataset: SAMM dataset instance
            casmeii_dataset: CASMEII dataset instance
            sample_ratio: Ratio of SAMM to total samples (0.5 = balanced)
        """
        self.samm_dataset = samm_dataset
        self.casmeii_dataset = casmeii_dataset
        self.sample_ratio = sample_ratio
        
        # Compute dataset sizes
        self.samm_size = len(samm_dataset) if samm_dataset else 0
        self.casmeii_size = len(casmeii_dataset) if casmeii_dataset else 0
        self.total_size = self.samm_size + self.casmeii_size
        
        if self.total_size == 0:
            raise ValueError("At least one dataset must be provided and non-empty")
    
    def __len__(self) -> int:
        """Get total dataset size."""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item (randomly sample from SAMM or CASMEII).
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with unified keys (handles both temporal and static data)
        """
        # Randomly decide which dataset to sample from
        if np.random.rand() < self.sample_ratio and self.samm_dataset:
            # Sample from SAMM (temporal)
            samm_idx = np.random.randint(0, self.samm_size)
            sample = self.samm_dataset[samm_idx]
            
            # Convert temporal to single frame (use middle frame)
            frames = sample['frames']  # [T, C, H, W]
            middle_frame = frames[frames.shape[0] // 2]  # [C, H, W]
            
            return {
                'image': middle_frame,
                'frames': frames,  # Keep full sequence for rPPG
                'emotion_label': sample['emotion_label'],
                'emotion_name': sample['emotion_name'],
                'source': 'samm',
                'is_temporal': True
            }
        
        elif self.casmeii_dataset:
            # Sample from CASMEII (static)
            casmeii_idx = np.random.randint(0, self.casmeii_size)
            sample = self.casmeii_dataset[casmeii_idx]
            
            return {
                'image': sample['image'],
                'frames': None,  # No temporal data
                'emotion_label': sample['emotion_label'],
                'emotion_name': sample['emotion_name'],
                'source': 'casmeii',
                'is_temporal': False
            }
        
        else:
            raise RuntimeError("No valid dataset available")


if __name__ == "__main__":
    # Test dataset loading
    print("=" * 80)
    print("CASMEII Dataset Loader - Validation Test")
    print("=" * 80)
    
    try:
        # Create datasets
        train_dataset = CASMEIIDataset(split='train')
        test_dataset = CASMEIIDataset(split='test')
        
        print(f"\n✅ Datasets loaded successfully!")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
        if len(train_dataset) > 0:
            # Get first sample
            sample = train_dataset[0]
            
            print(f"\n📊 Sample structure:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Emotion: {sample['emotion_name']} (idx={sample['emotion_label'].item()})")
            print(f"   Image ID: {sample['image_id']}")
            
            # Class distribution
            train_dist = train_dataset.get_class_distribution()
            test_dist = test_dataset.get_class_distribution()
            
            print(f"\n📈 Train emotion distribution:")
            for emotion, count in sorted(train_dist.items()):
                print(f"   {emotion:12s}: {count:4d} samples ({count/len(train_dataset)*100:.1f}%)")
            
            print(f"\n📈 Test emotion distribution:")
            for emotion, count in sorted(test_dist.items()):
                print(f"   {emotion:12s}: {count:4d} samples ({count/len(test_dataset)*100:.1f}%)")
            
            # Class weights
            weights = train_dataset.get_class_weights()
            print(f"\n⚖️ Class weights (for imbalanced training):")
            for i, (emotion, weight) in enumerate(zip(CASMEIIConfig.EMOTION_LABELS, weights)):
                print(f"   {emotion:12s}: {weight:.3f}")
        
        print(f"\n✅ All checks passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
