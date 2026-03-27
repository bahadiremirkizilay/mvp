"""
Landmark Cache System for Deterministic rPPG Processing
========================================================
Cache MediaPipe face landmarks to ensure reproducible results across
multiple validation runs.

Key Benefits:
- 100% deterministic validation results
- 2x faster processing (skips MediaPipe inference)
- Enables reliable model comparison and hyperparameter tuning

Cache Format:
- File: {video_path}.landmarks.npy
- Structure: dict with 'landmarks' (N×468×2) and 'metadata'
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2


def _compute_video_hash(video_path: Path, sample_frames: int = 10) -> str:
    """
    Compute a hash of the video file to detect changes.
    Samples first, middle, and last frames for efficiency.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    sample_frames : int
        Number of frames to sample for hash computation
        
    Returns
    -------
    str
        SHA256 hash (first 16 chars)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    hasher = hashlib.sha256()
    
    # Sample frames evenly throughout video
    sample_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            hasher.update(frame.tobytes())
    
    cap.release()
    return hasher.hexdigest()[:16]


def get_cache_path(video_path: Path) -> Path:
    """
    Get cache file path for a video.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
        
    Returns
    -------
    Path
        Cache file path (.landmarks.npy)
    """
    return video_path.with_suffix('.landmarks.npy')


def cache_exists(video_path: Path, check_hash: bool = True) -> bool:
    """
    Check if valid cache exists for video.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    check_hash : bool
        If True, verify video hasn't changed since cache creation
        
    Returns
    -------
    bool
        True if valid cache exists
    """
    cache_path = get_cache_path(video_path)
    
    if not cache_path.exists():
        return False
    
    if check_hash:
        try:
            cached_data = np.load(cache_path, allow_pickle=True).item()
            cached_hash = cached_data.get('metadata', {}).get('video_hash', '')
            current_hash = _compute_video_hash(video_path)
            
            if cached_hash != current_hash:
                print(f"  [WARN] Cache invalidated: video has changed")
                return False
        except Exception as e:
            print(f"  [WARN] Cache validation failed: {e}")
            return False
    
    return True


def load_cached_landmarks(video_path: Path) -> Optional[np.ndarray]:
    """
    Load cached landmarks from file.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
        
    Returns
    -------
    np.ndarray or None
        Landmark array (N_frames × 468 × 2) or None if cache invalid
    """
    cache_path = get_cache_path(video_path)
    
    try:
        cached_data = np.load(cache_path, allow_pickle=True).item()
        landmarks = cached_data['landmarks']
        
        print(f"  [CACHE] Loaded {len(landmarks)} frames from cache")
        print(f"  [CACHE] Cache created: {cached_data['metadata'].get('created_at', 'unknown')}")
        
        return landmarks
    except Exception as e:
        print(f"  [ERROR] Failed to load cache: {e}")
        return None


def save_landmarks_to_cache(
    video_path: Path,
    landmarks: np.ndarray,
    metadata: Optional[dict] = None
) -> None:
    """
    Save landmarks to cache file.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    landmarks : np.ndarray
        Landmark array (N_frames × 468 × 2)
    metadata : dict, optional
        Additional metadata to store
    """
    from datetime import datetime
    
    cache_path = get_cache_path(video_path)
    
    # Build metadata
    cache_metadata = {
        'video_path': str(video_path),
        'video_hash': _compute_video_hash(video_path),
        'created_at': datetime.now().isoformat(),
        'num_frames': len(landmarks),
        'landmark_shape': landmarks.shape,
    }
    
    if metadata:
        cache_metadata.update(metadata)
    
    # Save as single dict
    cache_data = {
        'landmarks': landmarks,
        'metadata': cache_metadata,
    }
    
    np.save(cache_path, cache_data)
    print(f"  [CACHE] Saved {len(landmarks)} frames to {cache_path.name}")


def extract_and_cache_landmarks(
    video_path: Path,
    extractor,
    force_recompute: bool = False,
    progress_callback = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Extract landmarks from video, using cache if available.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    extractor : ROIExtractor
        Initialized ROI extractor with MediaPipe
    force_recompute : bool
        If True, ignore cache and recompute landmarks
    progress_callback : callable, optional
        Function called with (frame_idx, total_frames)
        
    Returns
    -------
    landmarks : np.ndarray
        Landmark array (N_frames × 468 × 2)
    timestamps : list of float
        Frame timestamps (seconds)
    """
    video_path = Path(video_path)
    
    # Check cache
    if not force_recompute and cache_exists(video_path):
        landmarks = load_cached_landmarks(video_path)
        if landmarks is not None:
            # Reconstruct timestamps
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            timestamps = [i / fps for i in range(len(landmarks))]
            return landmarks, timestamps
    
    # Extract landmarks from video
    print(f"  [CACHE] Extracting landmarks (first time)...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    landmarks_list = []
    timestamps = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract landmarks using MediaPipe
        h, w = frame.shape[:2]
        
        # Use extractor's internal process to get landmarks
        # We need to access the raw landmarks before ROI processing
        import mediapipe as mp
        rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_input)
        
        result = extractor.face_landmarker.detect(mp_image)
        
        if result.face_landmarks:
            face_lms = result.face_landmarks[0]
            # Convert to pixel coordinates
            lm_array = np.array([
                [lm.x * w, lm.y * h] for lm in face_lms
            ], dtype=np.float32)  # shape: (468, 2)
        else:
            # No face detected - use zeros
            lm_array = np.zeros((468, 2), dtype=np.float32)
        
        landmarks_list.append(lm_array)
        timestamps.append(frame_idx / fps)
        frame_idx += 1
        
        if progress_callback:
            progress_callback(frame_idx, total_frames)
    
    cap.release()
    
    # Convert to numpy array
    landmarks = np.array(landmarks_list)  # shape: (N, 468, 2)
    
    # Save to cache
    save_landmarks_to_cache(video_path, landmarks, metadata={'fps': fps})
    
    return landmarks, timestamps


def clear_cache(video_path: Path) -> None:
    """
    Delete cache file for a video.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    """
    cache_path = get_cache_path(video_path)
    
    if cache_path.exists():
        cache_path.unlink()
        print(f"  [CACHE] Deleted cache: {cache_path.name}")
    else:
        print(f"  [CACHE] No cache to delete")


def clear_all_caches(dataset_path: Path) -> int:
    """
    Delete all cache files in a dataset directory.
    
    Parameters
    ----------
    dataset_path : Path
        Path to dataset root directory
        
    Returns
    -------
    int
        Number of cache files deleted
    """
    cache_files = list(dataset_path.rglob('*.landmarks.npy'))
    
    for cache_file in cache_files:
        cache_file.unlink()
    
    print(f"  [CACHE] Deleted {len(cache_files)} cache file(s)")
    return len(cache_files)
