#!/usr/bin/env python3
"""
Build deception manifest for Real-life Deception Detection 2016 dataset.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import cv2
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


def load_gesture_annotations(annotation_csv: str) -> Dict[str, Dict]:
    """Load gesture annotation CSV and return dict keyed by normalized video_id."""
    try:
        df = pd.read_csv(annotation_csv)
        
        # First column is 'id' (video name with .mp4)
        id_col = df.columns[0]
        
        gesture_dict = {}
        for _, row in df.iterrows():
            video_id = row[id_col]
            # Remove .mp4 extension if present
            video_id_normalized = str(video_id).replace('.mp4', '') if video_id else str(video_id)
            
            # All columns except first are gesture features
            gesture_features = row[1:].to_dict()
            gesture_dict[video_id_normalized] = gesture_features
        
        return gesture_dict
    except Exception as e:
        print(f"Error loading gesture annotations: {e}")
        return {}


def find_transcription(video_id: str, transcription_root: str) -> Tuple[bool, str]:
    """Check if transcription file exists for this video_id."""
    for label_dir in ['Deceptive', 'Truthful']:
        txt_path = os.path.join(transcription_root, label_dir, f"{video_id}.txt")
        if os.path.exists(txt_path):
            return True, txt_path
    return False, ""


def build_manifest(
    clips_root: str,
    annotation_csv: str,
    transcription_root: str,
    output_csv: str,
    min_duration_sec: float = 0,
    min_fps: float = 0
):
    """Build manifest CSV for Real-life 2016 deception dataset."""
    
    # Load gesture annotations
    print(f"Loading gesture annotations from {annotation_csv}...")
    gesture_dict = load_gesture_annotations(annotation_csv)
    print(f"  Loaded {len(gesture_dict)} gesture annotations")
    
    # Get gesture feature column names
    gesture_columns = list(next(iter(gesture_dict.values())).keys()) if gesture_dict else []
    print(f"  Gesture features: {gesture_columns[:5]}... ({len(gesture_columns)} total)")
    
    # Collect all videos
    rows = []
    deceptive_path = os.path.join(clips_root, "Deceptive")
    truthful_path = os.path.join(clips_root, "Truthful")
    
    print(f"\nScanning Deceptive folder...")
    for video_file in sorted(os.listdir(deceptive_path)):
        if not video_file.endswith(".mp4"):
            continue
        
        video_id = video_file.replace(".mp4", "")
        video_path = os.path.join(deceptive_path, video_file)
        
        # Get video metadata
        duration, fps, num_frames = get_video_metadata(video_path)
        if duration is None:
            print(f"  ⚠️  {video_id}: failed to read metadata, skipping")
            continue
        
        # Check quality filters
        quality_flags = []
        if duration < min_duration_sec:
            quality_flags.append(f"too_short_{duration:.1f}s")
        if fps < min_fps:
            quality_flags.append(f"low_fps_{fps:.1f}")
        
        # Get gesture features
        gesture_features = gesture_dict.get(video_id, {})
        
        # Check for transcription
        has_transcription, transcription_path = find_transcription(video_id, transcription_root)
        
        # Build row
        row = {
            'video_id': video_id,
            'video_path': video_path,
            'label': 1,  # 1 = lie/deceptive
            'label_name': 'lie',
            'duration_sec': round(duration, 2),
            'fps': round(fps, 2),
            'num_frames': num_frames,
            'has_transcription': has_transcription,
            'transcription_path': transcription_path if has_transcription else '',
            'quality_flags': ';'.join(quality_flags) if quality_flags else 'ok',
        }
        
        # Add gesture features
        row.update(gesture_features)
        rows.append(row)
    
    print(f"  Found {len(rows)} deceptive videos")
    
    print(f"\nScanning Truthful folder...")
    truthful_count = 0
    for video_file in sorted(os.listdir(truthful_path)):
        if not video_file.endswith(".mp4"):
            continue
        
        video_id = video_file.replace(".mp4", "")
        video_path = os.path.join(truthful_path, video_file)
        
        # Get video metadata
        duration, fps, num_frames = get_video_metadata(video_path)
        if duration is None:
            print(f"  ⚠️  {video_id}: failed to read metadata, skipping")
            continue
        
        # Check quality filters
        quality_flags = []
        if duration < min_duration_sec:
            quality_flags.append(f"too_short_{duration:.1f}s")
        if fps < min_fps:
            quality_flags.append(f"low_fps_{fps:.1f}")
        
        # Get gesture features
        gesture_features = gesture_dict.get(video_id, {})
        
        # Check for transcription
        has_transcription, transcription_path = find_transcription(video_id, transcription_root)
        
        # Build row
        row = {
            'video_id': video_id,
            'video_path': video_path,
            'label': 0,  # 0 = truth/truthful
            'label_name': 'truth',
            'duration_sec': round(duration, 2),
            'fps': round(fps, 2),
            'num_frames': num_frames,
            'has_transcription': has_transcription,
            'transcription_path': transcription_path if has_transcription else '',
            'quality_flags': ';'.join(quality_flags) if quality_flags else 'ok',
        }
        
        # Add gesture features
        row.update(gesture_features)
        rows.append(row)
        truthful_count += 1
    
    print(f"  Found {truthful_count} truthful videos")
    
    # Write manifest CSV
    print(f"\nWriting manifest to {output_csv}...")
    if not rows:
        print("  ERROR: No rows to write!")
        return False
    
    # Get all column names in order
    column_names = [
        'video_id', 'video_path', 'label', 'label_name',
        'duration_sec', 'fps', 'num_frames',
        'has_transcription', 'transcription_path',
        'quality_flags'
    ] + gesture_columns
    
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
    
    deceptive_count = sum(1 for r in rows if r['label'] == 1)
    truthful_count = sum(1 for r in rows if r['label'] == 0)
    print(f"  Deceptive (lies): {deceptive_count}")
    print(f"  Truthful: {truthful_count}")
    
    # Quality check
    ok_count = sum(1 for r in rows if r['quality_flags'] == 'ok')
    print(f"\nQuality:")
    print(f"  OK: {ok_count}")
    print(f"  Issues: {len(rows) - ok_count}")
    
    # Gestures check
    print(f"\nGesture features: {len(gesture_columns)}")
    gesture_coverage = sum(1 for r in rows if any(r.get(g) for g in gesture_columns if g))
    print(f"  Videos with gesture data: {gesture_coverage}/{len(rows)}")
    
    # Transcription check
    transcription_count = sum(1 for r in rows if r['has_transcription'])
    print(f"\nTranscriptions:")
    print(f"  Available: {transcription_count}/{len(rows)}")
    
    print(f"\n✅ Manifest written to: {output_csv}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deception manifest for Real-life 2016 dataset")
    parser.add_argument("--clips_root", default=r"data\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016\Clips",
                        help="Path to Clips folder")
    parser.add_argument("--annotation_csv", default=r"data\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016\Annotation\All_Gestures_Deceptive and Truthful.csv",
                        help="Path to gesture annotation CSV")
    parser.add_argument("--transcription_root", default=r"data\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016\Transcription",
                        help="Path to Transcription folder")
    parser.add_argument("--output_csv", default=r"data\RealLifeDeceptionDetection.2016\deception_manifest.csv",
                        help="Output manifest CSV path")
    parser.add_argument("--min_duration_sec", type=float, default=0,
                        help="Minimum video duration (seconds)")
    parser.add_argument("--min_fps", type=float, default=0,
                        help="Minimum frames per second")
    
    args = parser.parse_args()
    
    success = build_manifest(
        clips_root=args.clips_root,
        annotation_csv=args.annotation_csv,
        transcription_root=args.transcription_root,
        output_csv=args.output_csv,
        min_duration_sec=args.min_duration_sec,
        min_fps=args.min_fps
    )
    
    exit(0 if success else 1)
