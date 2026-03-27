#!/usr/bin/env python3
"""
Build deception manifest for Bag of Lies dataset.
"""

import os
import csv
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import cv2
import scipy.io.wavfile as wavfile
import argparse


def get_video_metadata(video_path: str) -> Tuple[float, float, int]:
    """Extract duration, fps, frame count from video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = num_frames / fps if fps > 0 else 0
        cap.release()
        
        return duration, fps, num_frames
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None, None


def get_audio_metadata(audio_path: str) -> Tuple[float, int]:
    """Extract duration and sample rate from WAV file."""
    try:
        sample_rate, audio_data = wavfile.read(audio_path)
        duration = len(audio_data) / sample_rate
        return duration, sample_rate
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None, None


def load_labels(labels_file: str) -> Dict[str, Tuple[str, bool]]:
    """Load labels from lie_detection_wav.txt file.
    Returns dict: {audio_filename: (gender, is_truth)}
    """
    labels = {}
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    audio_file = parts[0]  # e.g., "Ronda_1_Adri_audio.wav"
                    gender = parts[1].lower()
                    is_truth = parts[2].lower() == 'true'
                    labels[audio_file] = (gender, is_truth)
    except Exception as e:
        print(f"Error loading labels from {labels_file}: {e}")
    
    return labels


def build_manifest(
    input_dir: str,
    labels_file: str,
    output_csv: str,
    min_duration_sec: float = 0,
    min_fps: float = 0
):
    """Build manifest CSV for Bag of Lies deception dataset."""
    
    # Load labels
    print(f"Loading labels from {labels_file}...")
    labels = load_labels(labels_file)
    print(f"  Loaded {len(labels)} label entries")
    
    # Collect all videos
    rows = []
    
    print(f"\nScanning {input_dir} folder...")
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
    
    for video_file in video_files:
        # Extract base name (e.g., "Ronda_1_Adri")
        base_name = video_file.replace('.mp4', '')
        audio_file_name = f"{base_name}_audio.wav"
        
        # Get file paths
        video_path = os.path.join(input_dir, video_file)
        audio_path = os.path.join(input_dir, audio_file_name)
        
        # Check if both files exist
        if not os.path.exists(audio_path):
            print(f"  ⚠️  {base_name}: audio file not found, skipping")
            continue
        
        # Get video metadata
        duration, fps, num_frames = get_video_metadata(video_path)
        if duration is None:
            print(f"  ⚠️  {base_name}: failed to read video metadata, skipping")
            continue
        
        # Get audio metadata
        audio_duration, sample_rate = get_audio_metadata(audio_path)
        if audio_duration is None:
            print(f"  ⚠️  {base_name}: failed to read audio metadata, skipping")
            continue
        
        # Get label
        if audio_file_name not in labels:
            print(f"  ⚠️  {base_name}: label not found, skipping")
            continue
        
        gender, is_truth = labels[audio_file_name]
        
        # Check quality filters
        quality_flags = []
        if duration < min_duration_sec:
            quality_flags.append(f"video_too_short_{duration:.1f}s")
        if fps < min_fps:
            quality_flags.append(f"video_low_fps_{fps:.1f}")
        if audio_duration is not None and abs(duration - audio_duration) > 0.5:
            quality_flags.append(f"av_sync_mismatch")
        
        # Parse video_id components
        # Format: Ronda_X_SubjectName
        parts = base_name.split('_')
        if len(parts) >= 3:
            round_num = parts[0]  # "Ronda"
            round_idx = parts[1]  # "1", "2", ...
            subject_id = '_'.join(parts[2:])  # "Adri", "Dario", "Maria", etc.
        else:
            round_num = "unknown"
            round_idx = "unknown"
            subject_id = base_name
        
        # Build row
        row = {
            'video_id': base_name,
            'round': round_num,
            'round_idx': round_idx,
            'subject_id': subject_id,
            'gender': gender,
            'video_path': video_path,
            'audio_path': audio_path,
            'label': 0 if is_truth else 1,  # 0=truth, 1=lie
            'label_name': 'truth' if is_truth else 'lie',
            'duration_sec': round(duration, 2),
            'fps': round(fps, 2),
            'num_frames': num_frames,
            'audio_duration_sec': round(audio_duration, 2),
            'audio_sample_rate': sample_rate,
            'quality_flags': ';'.join(quality_flags) if quality_flags else 'ok',
        }
        
        rows.append(row)
    
    print(f"  Found {len(rows)} valid videos with audio pairs")
    
    # Write manifest CSV
    print(f"\nWriting manifest to {output_csv}...")
    if not rows:
        print("  ERROR: No rows to write!")
        return False
    
    column_names = [
        'video_id', 'round', 'round_idx', 'subject_id', 'gender',
        'video_path', 'audio_path',
        'label', 'label_name',
        'duration_sec', 'fps', 'num_frames',
        'audio_duration_sec', 'audio_sample_rate',
        'quality_flags'
    ]
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"MANIFEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(rows)}")
    
    truth_count = sum(1 for r in rows if r['label'] == 0)
    lie_count = sum(1 for r in rows if r['label'] == 1)
    print(f"  Truth: {truth_count}")
    print(f"  Lies: {lie_count}")
    print(f"  Imbalance ratio: 1:{lie_count/truth_count:.2f}" if truth_count > 0 else "  Imbalance: undefined")
    
    # Subject breakdown
    subjects = set(r['subject_id'] for r in rows)
    print(f"\nSubjects: {len(subjects)}")
    for subject in sorted(subjects):
        subject_rows = [r for r in rows if r['subject_id'] == subject]
        subject_truth = sum(1 for r in subject_rows if r['label'] == 0)
        subject_lie = sum(1 for r in subject_rows if r['label'] == 1)
        print(f"  {subject}: {len(subject_rows)} videos (T:{subject_truth} L:{subject_lie})")
    
    # Quality check
    ok_count = sum(1 for r in rows if r['quality_flags'] == 'ok')
    print(f"\nQuality:")
    print(f"  OK: {ok_count}")
    print(f"  Issues: {len(rows) - ok_count}")
    
    # Audio check
    audio_ok = sum(1 for r in rows if r['audio_path'] and os.path.exists(r['audio_path']))
    print(f"\nAudio:")
    print(f"  Files found: {audio_ok}/{len(rows)}")
    
    print(f"\n✅ Manifest written to: {output_csv}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deception manifest for Bag of Lies dataset")
    parser.add_argument("--input_dir", default=r"data\boxoflies",
                        help="Path to Bag of Lies data folder")
    parser.add_argument("--labels_file", default=r"data\boxoflies\lie_detection_wav.txt",
                        help="Path to labels file")
    parser.add_argument("--output_csv", default=r"data\boxoflies\deception_manifest.csv",
                        help="Output manifest CSV path")
    parser.add_argument("--min_duration_sec", type=float, default=0,
                        help="Minimum video duration (seconds)")
    parser.add_argument("--min_fps", type=float, default=0,
                        help="Minimum frames per second")
    
    args = parser.parse_args()
    
    success = build_manifest(
        input_dir=args.input_dir,
        labels_file=args.labels_file,
        output_csv=args.output_csv,
        min_duration_sec=args.min_duration_sec,
        min_fps=args.min_fps
    )
    
    exit(0 if success else 1)
