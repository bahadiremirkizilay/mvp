#!/usr/bin/env python3
"""
Unified dataset loader for CASME II + SAMM.
Combines both datasets for improved training.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

from emotion.samm_dataset import SAMMDataset
from emotion.casmeii_dataset import CASMEIIDataset
from emotion.augmentation import get_augmentation_pipeline


class UnifiedMicroExpressionDataset(Dataset):
    """
    Unified dataset combining SAMM and CASME II for micro-expression recognition.
    
    Features:
    - Combines temporal (SAMM) and static (CASMEII converted to temporal) data
    - Unified emotion labels across datasets
    - Data augmentation support
    - Balanced sampling

    Args:
        split: 'train', 'val', or 'test'
        use_samm: Whether to include SAMM dataset
        use_casmeii: Whether to include CASMEII dataset
        sequence_length: Number of frames in sequences
        convert_casmeii_to_temporal: Convert static CASMEII images to sequences
        augmentation_mode: 'train' or 'val' (controls augmentation)
        balance_classes: Whether to balance emotion classes
    """
    
    def __init__(
        self,
        split: str = 'train',
        use_samm: bool = True,
        use_casmeii: bool = True,
        sequence_length: int = 16,
        convert_casmeii_to_temporal: bool = True,
        augmentation_mode: Optional[str] = None,
        balance_classes: bool = True
    ):
        self.split = split
        self.sequence_length = sequence_length
        self.convert_casmeii_to_temporal = convert_casmeii_to_temporal
        
        # Auto-set augmentation mode
        if augmentation_mode is None:
            augmentation_mode = 'train' if split == 'train' else 'val'
        self.augmentation_mode = augmentation_mode
        
        # Load datasets
        self.datasets = []
        self.samples = []
        
        if use_samm:
            self._load_samm()
        
        if use_casmeii:
            self._load_casmeii()
        
        # Setup augmentation
        self.augmentation = get_augmentation_pipeline(
            mode=self.augmentation_mode,
            dataset_type='temporal'
        )
        
        # Class balancing
        self.balance_classes = balance_classes
        if balance_classes and split == 'train':
            self._compute_sample_weights()
        
        print(f"✅ Unified Dataset ({split}): {len(self.samples)} samples")
        print(f"   SAMM: {self.samm_count if hasattr(self, 'samm_count') else 0}")
        print(f"   CASMEII: {self.casmeii_count if hasattr(self, 'casmeii_count') else 0}")
    
    def _load_samm(self):
        """Load SAMM temporal sequences."""
        try:
            # Determine split
            if self.split == 'train':
                samm_split = 'train'
            elif self.split == 'val':
                samm_split = 'val'
            else:
                samm_split = 'test'
            
            samm_dataset = SAMMDataset(split=samm_split)
            
            # Add SAMM samples
            for idx in range(len(samm_dataset)):
                self.samples.append({
                    'source': 'samm',
                    'dataset': samm_dataset,
                    'idx': idx,
                    'type': 'temporal'
                })
            
            self.samm_count = len(samm_dataset)
            print(f"   Loaded SAMM: {self.samm_count} sequences")
            
        except Exception as e:
            print(f"   ⚠️ Failed to load SAMM: {e}")
            self.samm_count = 0
    
    def _load_casmeii(self):
        """Load CASMEII and optionally convert to temporal."""
        try:
            # Determine split
            if self.split == 'train':
                casmeii_split = 'train'
            else:
                casmeii_split = 'test'  # CASMEII has train/test split
            
            casmeii_dataset = CASMEIIDataset(
                split=casmeii_split,
                balance_classes=False  # We'll handle balancing globally
            )
            
            # Add CASMEII samples
            for idx in range(len(casmeii_dataset)):
                self.samples.append({
                    'source': 'casmeii',
                    'dataset': casmeii_dataset,
                    'idx': idx,
                    'type': 'static'  # Will convert to temporal if needed
                })
            
            self.casmeii_count = len(casmeii_dataset)
            print(f"   Loaded CASMEII: {self.casmeii_count} images")
            
        except Exception as e:
            print(f"   ⚠️ Failed to load CASMEII: {e}")
            self.casmeii_count = 0
    
    def _convert_static_to_temporal(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert a single image to a temporal sequence.
        
        Strategy: Duplicate image with slight variations to create sequence.
        This allows using static images in temporal models.
        """
        # Duplicate image
        sequence = [image] * self.sequence_length
        
        # Add slight variations (noise, jitter) to make it more realistic
        if self.augmentation_mode == 'train':
            varied_sequence = []
            for frame in sequence:
                # Add small Gaussian noise
                noise = torch.randn_like(frame) * 0.02
                varied_frame = frame + noise
                varied_frame = torch.clamp(varied_frame, -2.5, 2.5)  # Keep in normalized range
                varied_sequence.append(varied_frame)
            return varied_sequence
        
        return sequence
    
    def _compute_sample_weights(self):
        """Compute sample weights for balanced sampling (optimized version)."""
        # Get emotion labels efficiently from dataset samples (no loading images)
        emotion_labels = []
        for sample in self.samples:
            dataset = sample['dataset']
            idx = sample['idx']
            # Access the dataset's internal samples list directly (much faster!)
            emotion_labels.append(dataset.samples[idx]['emotion_idx'])
        
        emotion_labels = np.array(emotion_labels)
        
        # Compute class weights
        class_counts = np.bincount(emotion_labels, minlength=8)
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        class_weights = len(emotion_labels) / (8 * class_counts)
        
        # Compute sample weights
        self.sample_weights = torch.FloatTensor([
            class_weights[label] for label in emotion_labels
        ])
        
        print(f"   Class weights computed: {class_weights}")
    
    def get_sampler(self):
        """Get weighted sampler for balanced training."""
        if not self.balance_classes or self.split != 'train':
            return None
        
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the unified dataset.
        
        Returns:
            Dictionary with keys:
            - frames: Tensor of shape [T, C, H, W] (temporal sequence)
            - emotion_idx: Integer emotion label
            - emotion_name: String emotion name
            - source: 'samm' or 'casmeii'
        """
        sample_info = self.samples[idx]
        dataset = sample_info['dataset']
        sample_idx = sample_info['idx']
        
        # Get data from source dataset
        data = dataset[sample_idx]
        
        # Handle different data types
        if sample_info['type'] == 'temporal':
            # Already temporal sequence from SAMM
            frames = data['frames']  # [T, C, H, W]
        else:
            # Static image from CASMEII - convert to temporal
            if self.convert_casmeii_to_temporal:
                image = data['image']  # [C, H, W]
                frames = self._convert_static_to_temporal(image)
                frames = torch.stack(frames, dim=0)  # [T, C, H, W]
            else:
                # Return as single-frame sequence
                frames = data['image'].unsqueeze(0)  # [1, C, H, W]
        
        # Return unified format
        return {
            'frames': frames,
            'emotion_idx': data['emotion_label'],  # SAMM/CASMEII return 'emotion_label' as tensor
            'emotion_name': data['emotion_name'],
            'source': sample_info['source']
        }


def create_unified_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    use_samm: bool = True,
    use_casmeii: bool = True,
    sequence_length: int = 16,
    balance_classes: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for unified dataset.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_samm: Include SAMM dataset
        use_casmeii: Include CASMEII dataset
        sequence_length: Sequence length for temporal models
        balance_classes: Use weighted sampling for class balance
    
    Returns:
        (train_loader, val_loader)
    """
    print("=" * 80)
    print("CREATING UNIFIED DATALOADERS")
    print("=" * 80)
    
    # Create datasets
    train_dataset = UnifiedMicroExpressionDataset(
        split='train',
        use_samm=use_samm,
        use_casmeii=use_casmeii,
        sequence_length=sequence_length,
        augmentation_mode='train',
        balance_classes=balance_classes
    )
    
    val_dataset = UnifiedMicroExpressionDataset(
        split='val',
        use_samm=use_samm,
        use_casmeii=use_casmeii,
        sequence_length=sequence_length,
        augmentation_mode='val',
        balance_classes=False
    )
    
    # Create samplers
    train_sampler = train_dataset.get_sampler()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("=" * 80)
    print(f"✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print("=" * 80)
    
    return train_loader, val_loader


# Test code
if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED DATASET TEST")
    print("=" * 80)
    
    # Test dataset creation
    dataset = UnifiedMicroExpressionDataset(
        split='train',
        use_samm=True,
        use_casmeii=True,
        sequence_length=16
    )
    
    print(f"\n✅ Dataset created: {len(dataset)} samples")
    
    # Test sample loading
    print("\n✅ Testing sample loading...")
    sample = dataset[0]
    print(f"   Frames shape: {sample['frames'].shape}")
    print(f"   Emotion: {sample['emotion_name']} (idx={sample['emotion_idx']})")
    print(f"   Source: {sample['source']}")
    
    # Test dataloader creation
    print("\n✅ Creating dataloaders...")
    train_loader, val_loader = create_unified_dataloaders(
        batch_size=4,
        num_workers=0,  # 0 for testing
        use_samm=True,
        use_casmeii=True
    )
    
    print("\n✅ Testing batch loading...")
    batch = next(iter(train_loader))
    print(f"   Batch frames shape: {batch['frames'].shape}")
    print(f"   Batch emotions: {batch['emotion_name']}")
    print(f"   Batch sources: {batch['source']}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
