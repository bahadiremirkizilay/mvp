#!/usr/bin/env python3
"""
Data augmentation for emotion recognition.
Includes spatial and temporal augmentation strategies.
"""

import torch
import torchvision.transforms as transforms
import random
import numpy as np
from typing import List, Tuple, Optional


class SpatialAugmentation:
    """Spatial augmentation for individual frames."""
    
    def __init__(self, mode: str = 'train'):
        """
        Initialize spatial augmentation.
        
        Args:
            mode: 'train' or 'val' - training uses augmentation, val doesn't
        """
        self.mode = mode
        
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomRotation(degrees=5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image):
        """Apply spatial augmentation to image."""
        return self.transform(image)


class TemporalAugmentation:
    """Temporal augmentation for video sequences."""
    
    def __init__(
        self,
        mode: str = 'train',
        temporal_jitter_prob: float = 0.3,
        frame_dropout_prob: float = 0.2,
        speed_variation_prob: float = 0.3
    ):
        """
        Initialize temporal augmentation.
        
        Args:
            mode: 'train' or 'val'
            temporal_jitter_prob: Probability of temporal jittering
            frame_dropout_prob: Probability of dropping frames
            speed_variation_prob: Probability of speed variation
        """
        self.mode = mode
        self.temporal_jitter_prob = temporal_jitter_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.speed_variation_prob = speed_variation_prob
    
    def temporal_jitter(self, frames: List, max_jitter: int = 2) -> List:
        """
        Randomly shift frame indices (simulates timing variations).
        
        Args:
            frames: List of frames
            max_jitter: Maximum jitter in frames
        
        Returns:
            Jittered frame list
        """
        if len(frames) <= 3:
            return frames
        
        jitter = random.randint(-max_jitter, max_jitter)
        if jitter == 0:
            return frames
        
        # Circular shift
        jittered = frames[jitter:] + frames[:jitter]
        return jittered
    
    def frame_dropout(self, frames: List, dropout_rate: float = 0.1) -> List:
        """
        Randomly drop some frames (forces model to be robust).
        
        Args:
            frames: List of frames
            dropout_rate: Fraction of frames to drop
        
        Returns:
            Frames with some dropped
        """
        if len(frames) <= 4:
            return frames
        
        n_keep = max(4, int(len(frames) * (1 - dropout_rate)))
        indices = sorted(random.sample(range(len(frames)), n_keep))
        return [frames[i] for i in indices]
    
    def speed_variation(self, frames: List, speed_factor: float = None) -> List:
        """
        Vary sequence speed (simulate different expression speeds).
        
        Args:
            frames: List of frames
            speed_factor: Speed multiplier (0.8-1.2), if None randomly chosen
        
        Returns:
            Resampled frames
        """
        if len(frames) <= 4:
            return frames
        
        if speed_factor is None:
            speed_factor = random.uniform(0.8, 1.2)
        
        original_len = len(frames)
        new_len = int(original_len * speed_factor)
        new_len = max(4, min(new_len, original_len * 2))
        
        # Resample indices
        indices = np.linspace(0, original_len - 1, new_len).astype(int)
        return [frames[i] for i in indices]
    
    def __call__(self, frames: List) -> List:
        """
        Apply temporal augmentation to frame sequence.
        
        Args:
            frames: List of frames
        
        Returns:
            Augmented frame sequence
        """
        if self.mode != 'train':
            return frames
        
        # Apply augmentations randomly
        if random.random() < self.temporal_jitter_prob:
            frames = self.temporal_jitter(frames)
        
        if random.random() < self.frame_dropout_prob:
            frames = self.frame_dropout(frames)
        
        if random.random() < self.speed_variation_prob:
            frames = self.speed_variation(frames)
        
        return frames


class MixUp:
    """MixUp augmentation for emotion classification."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        
        Returns:
            mixed_images, labels_a, labels_b, lambda_mix
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class AugmentationPipeline:
    """Complete augmentation pipeline for emotion recognition."""
    
    def __init__(
        self,
        mode: str = 'train',
        use_spatial: bool = True,
        use_temporal: bool = True,
        use_mixup: bool = False
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            mode: 'train' or 'val'
            use_spatial: Whether to use spatial augmentation
            use_temporal: Whether to use temporal augmentation
            use_mixup: Whether to use MixUp (for batch-level aug)
        """
        self.mode = mode
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_mixup = use_mixup
        
        if use_spatial:
            self.spatial_aug = SpatialAugmentation(mode=mode)
        
        if use_temporal:
            self.temporal_aug = TemporalAugmentation(mode=mode)
        
        if use_mixup and mode == 'train':
            self.mixup = MixUp(alpha=0.2)
    
    def augment_sequence(self, frames: List) -> List:
        """
        Augment a sequence of frames.
        
        Args:
            frames: List of PIL Images or numpy arrays
        
        Returns:
            List of augmented frames (tensors)
        """
        # Temporal augmentation (changes sequence)
        if self.use_temporal:
            frames = self.temporal_aug(frames)
        
        # Spatial augmentation (per frame)
        if self.use_spatial:
            frames = [self.spatial_aug(frame) for frame in frames]
        else:
            # Just convert to tensor
            to_tensor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            frames = [to_tensor(frame) for frame in frames]
        
        return frames


def get_augmentation_pipeline(
    mode: str = 'train',
    dataset_type: str = 'temporal'
) -> AugmentationPipeline:
    """
    Get augmentation pipeline for dataset.
    
    Args:
        mode: 'train' or 'val'
        dataset_type: 'temporal' or 'static'
    
    Returns:
        AugmentationPipeline instance
    """
    use_temporal = (dataset_type == 'temporal')
    
    return AugmentationPipeline(
        mode=mode,
        use_spatial=True,
        use_temporal=use_temporal,
        use_mixup=(mode == 'train')  # MixUp only for training
    )


# Test code
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    print("=" * 80)
    print("AUGMENTATION PIPELINE TEST")
    print("=" * 80)
    
    # Create dummy frames
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(16)
    ]
    
    # Test spatial augmentation
    print("\n✅ Testing Spatial Augmentation...")
    spatial_aug = SpatialAugmentation(mode='train')
    aug_frame = spatial_aug(dummy_frames[0])
    print(f"   Input: PIL Image (224, 224, 3)")
    print(f"   Output: Tensor {aug_frame.shape}")
    
    # Test temporal augmentation
    print("\n✅ Testing Temporal Augmentation...")
    temporal_aug = TemporalAugmentation(mode='train')
    aug_frames = temporal_aug(dummy_frames)
    print(f"   Input: 16 frames")
    print(f"   Output: {len(aug_frames)} frames (temporal variation)")
    
    # Test full pipeline
    print("\n✅ Testing Full Pipeline...")
    pipeline = get_augmentation_pipeline(mode='train', dataset_type='temporal')
    aug_sequence = pipeline.augment_sequence(dummy_frames)
    print(f"   Input: {len(dummy_frames)} PIL Images")
    print(f"   Output: {len(aug_sequence)} Tensors, each {aug_sequence[0].shape}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
