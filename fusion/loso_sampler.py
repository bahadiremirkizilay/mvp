from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class FoldSplit:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


class LOSOSampler:
    """
    Leave-one-sample-out fold splitter with label-stratified validation split.

    This is a pragmatic fold helper for datasets where subject IDs are missing
    and each fold uses one held-out video as test sample.
    """

    def __init__(
        self,
        labels: Sequence[int],
        fold_idx: int,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        if not labels:
            raise ValueError("labels cannot be empty")
        if not (0.0 < val_frac < 0.5):
            raise ValueError("val_frac must be between 0 and 0.5")

        self.labels = np.asarray(labels, dtype=np.int64)
        self.n = int(self.labels.shape[0])
        self.fold_idx = int(fold_idx) % self.n
        self.val_frac = float(val_frac)
        self.seed = int(seed)

    def get_fold_split(self) -> Dict[str, List[int]]:
        all_idx = np.arange(self.n, dtype=np.int64)
        test_idx = np.array([self.fold_idx], dtype=np.int64)
        rem_idx = all_idx[all_idx != self.fold_idx]

        rng = np.random.default_rng(self.seed + self.fold_idx)

        val_parts: List[np.ndarray] = []
        train_parts: List[np.ndarray] = []

        for cls in np.unique(self.labels):
            cls_idx = rem_idx[self.labels[rem_idx] == cls]
            rng.shuffle(cls_idx)

            if cls_idx.size == 0:
                continue

            n_val = max(1, int(round(cls_idx.size * self.val_frac)))
            n_val = min(n_val, cls_idx.size)

            val_parts.append(cls_idx[:n_val])
            train_parts.append(cls_idx[n_val:])

        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)

        # If stratified split consumed all training samples for tiny classes,
        # move one val sample back to train to keep optimization step valid.
        if train_idx.size == 0 and val_idx.size > 0:
            train_idx = val_idx[:1]
            val_idx = val_idx[1:]

        return {
            "train": sorted(train_idx.tolist()),
            "val": sorted(val_idx.tolist()),
            "test": test_idx.tolist(),
        }
