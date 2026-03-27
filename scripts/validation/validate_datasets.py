"""
Dataset Validation and Quality Assessment
==========================================
Comprehensive validation scripts for SAMM and CASMEII datasets.

Checks:
    • File structure integrity
    • Annotation completeness
    • Image quality metrics
    • Label distribution
    • Data corruption detection
    • Missing files
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List
from collections import Counter
import warnings


def validate_samm_dataset(root_dir: Path = Path("data/SAMM")) -> Dict:
    """
    Validate SAMM dataset structure and quality.
    
    Args:
        root_dir: Root directory of SAMM dataset
    
    Returns:
        Dictionary with validation results
    """
    print("=" * 80)
    print("SAMM Dataset Validation")
    print("=" * 80)
    
    results = {
        'status': 'unknown',
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check root directory
    if not root_dir.exists():
        results['status'] = 'failed'
        results['errors'].append(f"Root directory not found: {root_dir}")
        return results
    
    # Check annotation file
    annotation_file = root_dir / "SAMM_Micro_FACS_Codes_v2.xlsx"
    if not annotation_file.exists():
        results['errors'].append(f"Annotation file missing: {annotation_file}")
    else:
        print(f"\n✅ Annotation file found: {annotation_file.name}")
        
        # Load annotations
        try:
            df = pd.read_excel(annotation_file, engine='openpyxl')
            results['stats']['total_annotations'] = len(df)
            print(f"   Total annotations: {len(df)}")
        except Exception as e:
            results['errors'].append(f"Failed to read annotations: {e}")
    
    # Check subject folders
    subject_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    results['stats']['num_subjects'] = len(subject_dirs)
    print(f"\n✅ Found {len(subject_dirs)} subject folders")
    
    # Validate each subject
    total_sequences = 0
    total_frames = 0
    corrupted_images = []
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        video_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
        total_sequences += len(video_dirs)
        
        for video_dir in video_dirs:
            frames = list(video_dir.glob(f"{subject_id}_*.jpg"))
            total_frames += len(frames)
            
            # Check random frames for corruption
            if len(frames) > 0:
                sample_frame = frames[len(frames) // 2]
                img = cv2.imread(str(sample_frame))
                if img is None:
                    corrupted_images.append(str(sample_frame))
    
    results['stats']['total_sequences'] = total_sequences
    results['stats']['total_frames'] = total_frames
    results['stats']['corrupted_images'] = len(corrupted_images)
    
    print(f"   Total sequences: {total_sequences}")
    print(f"   Total frames: {total_frames:,}")
    
    if len(corrupted_images) > 0:
        results['warnings'].append(f"{len(corrupted_images)} corrupted images found")
        print(f"\n⚠️ {len(corrupted_images)} corrupted images detected")
    else:
        print(f"\n✅ All sampled images are valid")
    
    # Final status
    if len(results['errors']) == 0:
        results['status'] = 'passed'
        print("\n✅ SAMM dataset validation PASSED")
    else:
        results['status'] = 'failed'
        print("\n❌ SAMM dataset validation FAILED")
        for error in results['errors']:
            print(f"   • {error}")
    
    return results


def validate_casmeii_dataset(root_dir: Path = Path("data/CASMEII")) -> Dict:
    """
    Validate CASMEII dataset structure and quality.
    
    Args:
        root_dir: Root directory of CASMEII dataset
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 80)
    print("CASMEII Dataset Validation")
    print("=" * 80)
    
    results = {
        'status': 'unknown',
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check root directory
    if not root_dir.exists():
        results['status'] = 'failed'
        results['errors'].append(f"Root directory not found: {root_dir}")
        return results
    
    # Check train/test split
    train_dir = root_dir / "train"
    test_dir = root_dir / "test"
    
    if not train_dir.exists():
        results['errors'].append("Train directory missing")
    if not test_dir.exists():
        results['errors'].append("Test directory missing")
    
    # Validate train split
    train_stats = {}
    if train_dir.exists():
        print(f"\n📂 Validating train split...")
        emotion_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            images = list(emotion_dir.glob("*.jpg")) + \
                    list(emotion_dir.glob("*.jpeg")) + \
                    list(emotion_dir.glob("*.png"))
            train_stats[emotion] = len(images)
            print(f"   {emotion:12s}: {len(images):4d} images")
        
        results['stats']['train'] = train_stats
        results['stats']['train_total'] = sum(train_stats.values())
    
    # Validate test split
    test_stats = {}
    if test_dir.exists():
        print(f"\n📂 Validating test split...")
        emotion_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            images = list(emotion_dir.glob("*.jpg")) + \
                    list(emotion_dir.glob("*.jpeg")) + \
                    list(emotion_dir.glob("*.png"))
            test_stats[emotion] = len(images)
            print(f"   {emotion:12s}: {len(images):4d} images")
        
        results['stats']['test'] = test_stats
        results['stats']['test_total'] = sum(test_stats.values())
    
    # Check image quality
    print(f"\n🔍 Checking image quality...")
    corrupted_count = 0
    
    for split_dir in [train_dir, test_dir]:
        if not split_dir.exists():
            continue
        
        for emotion_dir in split_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            
            images = list(emotion_dir.glob("*.jpg"))[:5]  # Sample 5 images per emotion
            
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupted_count += 1
    
    results['stats']['corrupted_images'] = corrupted_count
    
    if corrupted_count == 0:
        print(f"   ✅ All sampled images are valid")
    else:
        print(f"   ⚠️ {corrupted_count} corrupted images detected")
        results['warnings'].append(f"{corrupted_count} corrupted images")
    
    # Final status
    if len(results['errors']) == 0:
        results['status'] = 'passed'
        print("\n✅ CASMEII dataset validation PASSED")
    else:
        results['status'] = 'failed'
        print("\n❌ CASMEII dataset validation FAILED")
        for error in results['errors']:
            print(f"   • {error}")
    
    return results


def validate_all_datasets():
    """Validate all datasets."""
    print("=" * 80)
    print("MULTIMODAL DATASET VALIDATION")
    print("=" * 80)
    
    results = {}
    
    # Validate SAMM
    results['samm'] = validate_samm_dataset()
    
    # Validate CASMEII
    results['casmeii'] = validate_casmeii_dataset()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = all(r['status'] == 'passed' for r in results.values())
    
    for dataset_name, dataset_results in results.items():
        status_icon = "✅" if dataset_results['status'] == 'passed' else "❌"
        print(f"\n{status_icon} {dataset_name.upper()}: {dataset_results['status'].upper()}")
        
        if dataset_results.get('stats'):
            for key, value in dataset_results['stats'].items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"   {key}: {value}")
    
    if all_passed:
        print("\n✅ ALL DATASETS VALIDATED SUCCESSFULLY")
        print("\n🚀 Ready to begin training!")
    else:
        print("\n⚠️ SOME DATASETS FAILED VALIDATION")
        print("   Please fix errors before training.")
    
    return results


if __name__ == "__main__":
    validate_all_datasets()
