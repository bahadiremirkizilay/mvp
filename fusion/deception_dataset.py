#!/usr/bin/env python3
"""DeceptionDataset - cache-backed dataset for deception training."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import hashlib
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DeceptionDataset(Dataset):
    """
    PyTorch Dataset for loading deception detection samples.
    
    Input: manifest CSV with video paths and labels
    Output: features dict + label
    
    Supports:
    - Real-life 2016 (141 gesture features)
    - Bag of Lies (audio + video)
    - Combined dataset (union of both)
    """
    
    def __init__(
        self,
        manifest_csv: str,
        feature_cache_dir: str = "checkpoints/feature_cache",
        modalities: List[str] = None,
        include_gesture_features: bool = True,
        feature_extractor=None,
        feature_mode: str = "smoke",
        cache_version: str = "v1",
        sequence_length: int = 20,
        verbose: bool = True,
        modality_dim_overrides: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            manifest_csv: Path to deception manifest CSV file
            feature_cache_dir: Where to cache extracted features
            modalities: ['rppg', 'emotion', 'behavioral', 'audio'] or None for all
            include_gesture_features: Include gesture annotations as features
            feature_extractor: Callable feature extractor or None
            sequence_length: Temporal length for fallback generated sequences
            verbose: Print progress messages
        """
        
        self.manifest_csv = manifest_csv
        self.feature_cache_dir = Path(feature_cache_dir)
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.modalities = modalities or ['rppg', 'emotion', 'behavioral']
        self.include_gesture_features = include_gesture_features
        self.feature_extractor = feature_extractor
        self.feature_mode = str(feature_mode).lower().strip()
        if self.feature_mode not in {"smoke", "real"}:
            raise ValueError("feature_mode must be one of: smoke, real")
        self.cache_version = str(cache_version).strip() or "v1"
        self.sequence_length = int(sequence_length)
        self.verbose = verbose
        self.default_dims = {
            'rppg': 8,
            'emotion': 16,
            'behavioral': 9,
            'audio': 24,   # 12 MFCCs + 12 delta-MFCCs (librosa extraction)
            'verbal': 12,  # linguistic statistics from transcript
            'gesture': 1,
        }
        # Allow caller to override dims (e.g. when CNN emotion model outputs 7 classes)
        if modality_dim_overrides:
            self.default_dims.update(modality_dim_overrides)
        
        # Load manifest
        if verbose:
            print(f"Loading manifest from {manifest_csv}...")
        self.manifest_df = pd.read_csv(manifest_csv)
        if verbose:
            print(f"  Loaded {len(self.manifest_df)} samples")
        
        # Validate required columns
        self._validate_manifest()
        
        # Identify gesture columns (if present)
        self.gesture_columns = self._identify_gesture_columns()
        self.default_dims['gesture'] = max(1, len(self.gesture_columns))
        if self.gesture_columns and verbose:
            print(f"  Found {len(self.gesture_columns)} gesture features")
        
        if verbose:
            print(f"✅ Dataset initialized with {len(self)} samples")
    
    def _validate_manifest(self):
        """Check manifest has required columns."""
        required = ['video_id', 'label', 'video_path']
        missing = [c for c in required if c not in self.manifest_df.columns]
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
    
    def _identify_gesture_columns(self) -> List[str]:
        """Find columns that are gesture features (binary 0/1)."""
        gesture_cols = []
        
        # Skip known non-gesture columns
        skip_patterns = ['video_', 'audio_', 'label', 'duration', 'fps', 'num_frames', 
                        'transcription', 'quality', 'subject', 'gender', 'round']
        
        for col in self.manifest_df.columns:
            # Skip known columns
            if any(pattern in col.lower() for pattern in skip_patterns):
                continue
            
            # Check if column is binary (0/1) or boolean-like
            try:
                unique_vals = self.manifest_df[col].dropna().unique()
                if len(unique_vals) <= 2 and all(v in [0, 1, '0', '1', True, False] for v in unique_vals):
                    gesture_cols.append(col)
            except:
                pass
        
        return gesture_cols
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.manifest_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one sample from dataset.
        
        Returns dict:
        {
            'video_id': str,
            'label': 0 or 1,
            'label_name': 'truth' or 'lie',
            'features': {
                'rppg': np.array (T, D_rppg),
                'emotion': np.array (T, D_emotion),
                'behavioral': np.array (T, D_behavioral),
                'audio': np.array (T, D_audio) or None,
                'gesture': np.array (D_gesture,) or None
            },
            'metadata': {
                'video_path': str,
                'duration_sec': float,
                'fps': float,
                ...
            }
        }
        """
        
        row = self.manifest_df.iloc[idx]
        video_id = row['video_id']
        label = int(row['label'])
        
        # Try to load cached features.
        cached_path = self._get_cache_path(video_id)
        if cached_path.exists():
            try:
                return self._load_cached_sample(cached_path, row, label)
            except (zipfile.BadZipFile, OSError, ValueError):
                # Cache can be interrupted/corrupted during long LOSO runs.
                # Regenerate sample once by deleting the bad artifact.
                try:
                    cached_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # Generate and cache on first access.
        sample = self._extract_and_cache_sample(cached_path, row, label)
        return sample

    def _get_cache_path(self, video_id: str) -> Path:
        """Build mode-aware cache file path to prevent stale feature reuse."""
        safe_id = str(video_id).replace("/", "_").replace("\\", "_")
        suffix = f"{self.feature_mode}_{self.cache_version}_T{self.sequence_length}"
        return self.feature_cache_dir / f"{safe_id}__{suffix}.npz"
    
    def _load_cached_sample(self, cached_path: Path, row, label: int) -> Dict:
        """Load pre-extracted features from cache."""
        cache = np.load(cached_path, allow_pickle=True)
        
        features = {}
        for mod in self.modalities:
            if f'{mod}_features' in cache:
                features[mod] = cache[f'{mod}_features']
            else:
                features[mod] = None
        
        features['gesture'] = self._gesture_vector(row)
        
        return {
            'video_id': str(row['video_id']),
            'label': label,
            'label_name': str(row.get('label_name', 'unknown')),
            'features': features,
            'metadata': self._extract_metadata(row)
        }

    def _extract_and_cache_sample(self, cached_path: Path, row, label: int) -> Dict:
        """Extract (or synthesize) features and persist in cache."""
        features = self._extract_features(row, label)

        to_save = {}
        for mod in self.modalities:
            val = features.get(mod)
            if val is not None:
                to_save[f'{mod}_features'] = val.astype(np.float32)
        np.savez_compressed(cached_path, **to_save)

        features['gesture'] = self._gesture_vector(row)
        return {
            'video_id': str(row['video_id']),
            'label': int(label),
            'label_name': str(row.get('label_name', 'unknown')),
            'features': features,
            'metadata': self._extract_metadata(row)
        }

    def _extract_features(self, row, label: int) -> Dict[str, Optional[np.ndarray]]:
        """Extract features via provided extractor, else deterministic fallback."""
        if callable(self.feature_extractor):
            try:
                extracted = self.feature_extractor(
                    video_path=str(row.get('video_path', '')),
                    audio_path=str(row.get('audio_path', '')) if 'audio_path' in row else None,
                    row=row,
                )
                if isinstance(extracted, dict):
                    normalized = {}
                    for mod in self.modalities:
                        arr = extracted.get(mod)
                        normalized[mod] = self._as_2d(arr, mod)
                    return normalized
            except Exception:
                pass

        return self._fallback_features(row, label, include_label_signal=(self.feature_mode == "smoke"))

    def _fallback_features(self, row, label: int, include_label_signal: bool = True) -> Dict[str, Optional[np.ndarray]]:
        """Generate deterministic sequence features from manifest metadata."""
        video_id = str(row.get('video_id', 'unknown'))
        seed = int(hashlib.md5(video_id.encode('utf-8')).hexdigest()[:8], 16)

        dur = float(row.get('duration_sec', 0.0) or 0.0)
        fps = float(row.get('fps', 0.0) or 0.0)
        nfrm = float(row.get('num_frames', 0.0) or 0.0)
        has_txt = float(bool(row.get('has_transcription', False)))
        has_aud = float('audio_path' in row and isinstance(row.get('audio_path', None), str) and len(str(row.get('audio_path', ''))) > 0)
        aud_dur = float(row.get('audio_duration_sec', 0.0) or 0.0)

        g = self._gesture_vector(row)
        if g is None or g.size == 0:
            g_mean, g_std = 0.0, 0.0
        else:
            g_mean = float(np.mean(g))
            g_std = float(np.std(g))

        feats: Dict[str, Optional[np.ndarray]] = {}
        label_sig = float(label) if include_label_signal else 0.0
        for i, mod in enumerate(self.modalities):
            if mod == 'rppg':
                base = np.array([
                    dur / 60.0, fps / 30.0, np.log1p(nfrm) / 10.0,
                    label_sig, g_mean, g_std, has_txt, 0.5 * label_sig + 0.5 * g_mean,
                ], dtype=np.float32)
            elif mod == 'emotion':
                base = np.zeros((self.default_dims['emotion'],), dtype=np.float32)
                base[:8] = np.array([label_sig, 1.0 - label_sig, g_mean, g_std, has_txt, dur / 60.0, fps / 30.0, 0.0], dtype=np.float32)
                if g is not None and g.size > 0:
                    k = min(g.size, 8)
                    base[8:8 + k] = g[:k]
            elif mod == 'behavioral':
                base = np.array([
                    g_mean,
                    g_std,
                    has_txt,
                    dur / 60.0,
                    fps / 30.0,
                    label_sig,
                    float(g.sum()) if g is not None else 0.0,
                    np.log1p(nfrm) / 10.0,
                    has_aud,
                ], dtype=np.float32)
            elif mod == 'audio':
                base = np.array([has_aud, aud_dur / 60.0, label_sig, g_mean, g_std, dur / 60.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif mod == 'verbal':
                base = np.zeros((self.default_dims['verbal'],), dtype=np.float32)
                base[0] = has_txt  # has transcription
            else:
                base = np.zeros((self.default_dims.get(mod, 4),), dtype=np.float32)

            feats[mod] = self._repeat_with_noise(base, self.sequence_length, 0.03, seed + (i + 1) * 97)

        return feats

    def _repeat_with_noise(self, vec: np.ndarray, t: int, noise_scale: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_scale, size=(t, vec.shape[0])).astype(np.float32)
        return np.tile(vec[None, :], (t, 1)).astype(np.float32) + noise

    def _as_2d(self, arr, mod: str) -> np.ndarray:
        """Normalize a raw modality array to [T, D] and expected D."""
        d = self.default_dims.get(mod, 4)
        if arr is None:
            return np.zeros((self.sequence_length, d), dtype=np.float32)

        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            x = np.zeros((1, d), dtype=np.float32)

        if x.shape[1] < d:
            pad = np.zeros((x.shape[0], d - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > d:
            x = x[:, :d]

        return x

    def _gesture_vector(self, row) -> Optional[np.ndarray]:
        if not self.include_gesture_features or not self.gesture_columns:
            return None
        vec = []
        for col in self.gesture_columns:
            try:
                vec.append(float(row.get(col, 0) or 0))
            except Exception:
                vec.append(0.0)
        return np.asarray(vec, dtype=np.float32)
    
    def _extract_metadata(self, row) -> Dict:
        """Extract metadata fields from row."""
        metadata = {
            'video_path': str(row.get('video_path', '')),
            'audio_path': str(row.get('audio_path', '')) if 'audio_path' in row else None,
        }
        
        # Optional fields
        for field in ['duration_sec', 'fps', 'num_frames', 'audio_duration_sec', 
                     'subject_id', 'gender', 'round_idx']:
            if field in row:
                try:
                    val = row[field]
                    if pd.notna(val):
                        metadata[field] = float(val) if field in ['duration_sec', 'fps'] else val
                except:
                    pass
        
        return metadata

    def build_tensor_pack(
        self,
        indices: List[int],
        modalities: Optional[List[str]] = None,
        target_length: int = 20,
        temporal_augment: bool = False,
        temporal_crop_frac: float = 0.7,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Build fixed-size [N, T, D] modality tensors for selected indices.

        Args:
            temporal_augment: If True, randomly crop a sub-window of each sample's
                sequence before resampling back to target_length.  This prevents the
                model from learning *when* in the video a signal occurs (position
                bias) and forces it to learn the *shape* of dynamics instead.
                Only enable for training splits — leave False for val/test.
            temporal_crop_frac: Fraction of T to keep in each crop (default 0.7).
        """
        use_modalities = modalities or self.modalities
        packed = {m: [] for m in use_modalities}
        labels: List[float] = []
        rng = np.random.default_rng()   # non-seeded for training randomness

        for idx in indices:
            s = self[idx]
            labels.append(float(s['label']))
            for m in use_modalities:
                arr = s['features'].get(m)
                arr2 = self._as_2d(arr, m)
                if temporal_augment and arr2.shape[0] > 1:
                    crop_len = max(2, int(arr2.shape[0] * temporal_crop_frac))
                    max_start = arr2.shape[0] - crop_len
                    if max_start > 0:
                        start = int(rng.integers(0, max_start + 1))
                        arr2 = arr2[start: start + crop_len]
                packed[m].append(self._resample_time(arr2, target_length))

        out = {m: np.stack(v, axis=0).astype(np.float32) for m, v in packed.items()}
        y = np.asarray(labels, dtype=np.float32)
        return out, y

    def _resample_time(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        if arr.shape[0] == target_length:
            return arr.astype(np.float32)
        if arr.shape[0] == 1:
            return np.repeat(arr, target_length, axis=0).astype(np.float32)

        src_t = np.linspace(0.0, 1.0, num=arr.shape[0], dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
        cols = [np.interp(dst_t, src_t, arr[:, j]).astype(np.float32) for j in range(arr.shape[1])]
        return np.stack(cols, axis=1)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get count of samples per label."""
        return dict(self.manifest_df['label'].value_counts())
    
    def get_subjects(self) -> List[str]:
        """Get unique subject IDs if available."""
        if 'subject_id' in self.manifest_df.columns:
            return sorted(self.manifest_df['subject_id'].unique().tolist())
        return []
    
    def get_stratified_split(self, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/val/test with label stratification.
        
        Args:
            train_frac: Fraction for training (default 70%)
            val_frac: Fraction for validation (default 15%)
            seed: Random seed
            
        Returns:
            (train_indices, val_indices, test_indices)
        """
        from sklearn.model_selection import train_test_split
        
        np.random.seed(seed)
        
        # Group by label
        label_groups = {}
        for label in self.manifest_df['label'].unique():
            indices = self.manifest_df[self.manifest_df['label'] == label].index.tolist()
            label_groups[label] = indices
        
        train_idx, val_idx, test_idx = [], [], []
        
        # Split each label group
        for label, indices in label_groups.items():
            train_set, temp = train_test_split(indices, train_size=train_frac, random_state=seed)
            val_set, test_set = train_test_split(temp, train_size=val_frac/(1-train_frac), random_state=seed)
            
            train_idx.extend(train_set)
            val_idx.extend(val_set)
            test_idx.extend(test_set)
        
        return sorted(train_idx), sorted(val_idx), sorted(test_idx)


class DeceptionDataLoader:
    """Convenience wrapper for creating DataLoaders."""
    
    @staticmethod
    def create_train_loader(
        dataset: DeceptionDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """Create training DataLoader."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DeceptionDataLoader._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """Collate variable-length sequences by zero-padding temporal axis."""
        video_ids = [b['video_id'] for b in batch]
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32).view(-1, 1)
        metadata = [b['metadata'] for b in batch]

        feature_keys = sorted({k for b in batch for k in b['features'].keys() if k != 'gesture'})
        features_out: Dict[str, torch.Tensor] = {}
        lengths_out: Dict[str, torch.Tensor] = {}

        for key in feature_keys:
            arrs = []
            lengths = []
            dim = None
            for b in batch:
                x = b['features'].get(key)
                if x is None:
                    continue
                x2 = np.asarray(x, dtype=np.float32)
                if x2.ndim == 1:
                    x2 = x2[None, :]
                if x2.ndim != 2:
                    continue
                arrs.append(x2)
                lengths.append(x2.shape[0])
                dim = x2.shape[1]

            if not arrs or dim is None:
                continue

            max_t = max(lengths)
            stacked = []
            for b in batch:
                x = b['features'].get(key)
                if x is None:
                    x2 = np.zeros((1, dim), dtype=np.float32)
                else:
                    x2 = np.asarray(x, dtype=np.float32)
                    if x2.ndim == 1:
                        x2 = x2[None, :]
                    if x2.shape[1] < dim:
                        pad_d = np.zeros((x2.shape[0], dim - x2.shape[1]), dtype=np.float32)
                        x2 = np.concatenate([x2, pad_d], axis=1)
                    elif x2.shape[1] > dim:
                        x2 = x2[:, :dim]

                if x2.shape[0] < max_t:
                    pad_t = np.zeros((max_t - x2.shape[0], dim), dtype=np.float32)
                    x2 = np.concatenate([x2, pad_t], axis=0)
                stacked.append(x2)

            features_out[key] = torch.tensor(np.stack(stacked, axis=0), dtype=torch.float32)
            lengths_out[key] = torch.tensor([np.asarray(b['features'].get(key)).shape[0] if b['features'].get(key) is not None else 1 for b in batch], dtype=torch.long)

        return {
            'video_ids': video_ids,
            'labels': labels,
            'features': features_out,
            'lengths': lengths_out,
            'metadata': metadata,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeceptionDataset")
    parser.add_argument("--manifest", default=r"data\RealLifeDeceptionDetection.2016\deception_manifest.csv",
                        help="Manifest CSV path")
    parser.add_argument("--num_samples", type=int, default=5, help="Samples to inspect")
    parser.add_argument("--cache_dir", default="checkpoints/feature_cache", help="Feature cache directory")
    
    args = parser.parse_args()
    
    # Load dataset
    print("\n" + "="*80)
    print("DECEPTION DATASET TEST")
    print("="*80 + "\n")
    
    dataset = DeceptionDataset(
        manifest_csv=args.manifest,
        feature_cache_dir=args.cache_dir,
        modalities=['rppg', 'emotion', 'behavioral'],
        include_gesture_features=True,
        verbose=True
    )
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Label distribution: {dataset.get_label_distribution()}")
    
    subjects = dataset.get_subjects()
    if subjects:
        print(f"  Subjects: {subjects}")
    
    # Test split
    train_idx, val_idx, test_idx = dataset.get_stratified_split()
    print(f"\n📋 Stratified Split:")
    print(f"  Train: {len(train_idx)} ({100*len(train_idx)/len(dataset):.1f}%)")
    print(f"  Val: {len(val_idx)} ({100*len(val_idx)/len(dataset):.1f}%)")
    print(f"  Test: {len(test_idx)} ({100*len(test_idx)/len(dataset):.1f}%)")
    
    # Inspect samples
    print(f"\n📸 Sample Inspection (first {args.num_samples} samples):")
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n  [{i}] {sample['video_id']}")
        print(f"      Label: {sample['label_name']} ({sample['label']})")
        print(f"      Duration: {sample['metadata'].get('duration_sec', 'N/A')} sec")
        mods = [m for m in sample['features'] if sample['features'][m] is not None]
        print(f"      Modalities: {mods}")
        for m in ['rppg', 'emotion', 'behavioral', 'audio']:
            if sample['features'].get(m) is not None:
                print(f"      {m} shape: {tuple(sample['features'][m].shape)}")
        print(f"      Gesture features: {'Yes' if sample['features'].get('gesture') is not None else 'No'}")
    
    print("\n✅ Dataset test complete!\n")
