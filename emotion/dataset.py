"""
Emotion Dataset Loader  [Phase 2 — Not yet implemented]
========================================================
Will provide a PyTorch Dataset class for loading and augmenting
AffectNet facial expression images with valence/arousal annotations.

Planned features:
    • AffectNetDataset(root, split, transform) → (image, label, va)
    • Weighted sampler for class-imbalance mitigation
    • Albumentations augmentation pipeline (flip, colour jitter, cutout)
    • CSV annotation parser for AffectNet 8-class and 2-class (VA) modes
"""

# TODO (Phase 2): Implement AffectNetDataset


class AffectNetDataset:
    """Placeholder — implement in Phase 2."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AffectNetDataset is scheduled for Phase 2."
        )
