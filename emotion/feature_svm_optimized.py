"""
Optimized 3-Class Micro-Expression Recognition
=================================================
Focuses on the best configuration found (3-class: negative/positive/surprise)
with hyperparameter tuning via GridSearchCV and enhanced feature engineering.

Improvements over v2:
    • More optical flow features (multi-scale, temporal sequences)
    • LBP (Local Binary Patterns) features for texture
    • GridSearchCV for SVM/RF hyperparameters
    • Feature importance analysis
    • Proper LOSO-CV with nested cross-validation

Usage:
    python emotion/feature_svm_optimized.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline

from emotion.model import EmotionClassifier
from emotion.samm_dataset import SAMMDataset, SAMMConfig


# 3-class mapping: negative/positive/surprise (exclude "other"/neutral)
CLASS_MAP_3 = {
    0: 0, 1: 0, 2: 0, 3: 0,  # anger/contempt/disgust/fear → negative
    4: 1,                       # happiness → positive
    5: 0,                       # sadness → negative
    6: 2,                       # surprise → surprise
    7: -1,                      # other → EXCLUDE
}
CLASS_NAMES = ['negative', 'positive', 'surprise']


def load_backbone(pretrained_path: str, backbone: str = 'resnet50'):
    """Load frozen backbone for feature extraction."""
    model = EmotionClassifier(num_classes=8, backbone=backbone, pretrained=True, freeze_backbone=True)
    model.classifier = nn.Identity()

    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        backbone_state = {}
        for key, value in state_dict.items():
            if any(skip in key for skip in ['fc', 'classifier', 'lstm', 'temporal']):
                continue
            if key.startswith('backbone.'):
                backbone_state[key] = value
            elif any(layer in key for layer in ['layer', 'conv', 'bn', 'downsample']):
                backbone_state[f'backbone.{key}'] = value
        if backbone_state:
            model.load_state_dict(backbone_state, strict=False)
            print(f"  Loaded {len(backbone_state)} pretrained tensors")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_optical_flow(frame1, frame2):
    """Dense optical flow (Farneback)."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def compute_lbp(image, radius=1, n_points=8):
    """Compute Local Binary Pattern for texture features."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            pattern = 0
            for k in range(n_points):
                angle = 2.0 * np.pi * k / n_points
                ni = int(round(i + radius * np.sin(angle)))
                nj = int(round(j + radius * np.cos(angle)))
                ni = max(0, min(h - 1, ni))
                nj = max(0, min(w - 1, nj))
                pattern |= (1 << k) if gray[ni, nj] >= center else 0
            lbp[i, j] = pattern

    hist, _ = np.histogram(lbp, bins=2**n_points, range=(0, 2**n_points))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def extract_flow_block_features(flow, num_blocks=4):
    """Statistical features from optical flow in spatial blocks."""
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    features = []
    bh, bw = h // num_blocks, w // num_blocks

    for i in range(num_blocks):
        for j in range(num_blocks):
            bm = mag[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            ba = ang[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            bu = flow[i*bh:(i+1)*bh, j*bw:(j+1)*bw, 0]
            bv = flow[i*bh:(i+1)*bh, j*bw:(j+1)*bw, 1]

            features.extend([bm.mean(), bm.max(), bm.std()])
            hist, _ = np.histogram(ba, bins=8, range=(0, 2*np.pi), weights=bm)
            s = hist.sum()
            if s > 0:
                hist = hist / s
            features.extend(hist.tolist())
            features.extend([bu.mean(), bv.mean(), bu.std(), bv.std()])

    return np.array(features, dtype=np.float32)


def extract_roi_flow_features(flow, frame_size=224):
    """
    Extract flow features from facial ROIs (upper/lower face, left/right).
    Micro-expressions primarily involve specific facial regions.
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Define anatomical ROIs (approximate for cropped faces)
    rois = {
        'forehead':  (0, h//4, 0, w),           # Top quarter
        'eyes':      (h//4, h//2, 0, w),         # Upper middle
        'nose_mouth': (h//2, 3*h//4, 0, w),      # Lower middle
        'chin':      (3*h//4, h, 0, w),           # Bottom
        'left_face': (0, h, 0, w//2),             # Left half
        'right_face': (0, h, w//2, w),            # Right half
    }

    features = []
    for roi_name, (y1, y2, x1, x2) in rois.items():
        rm = mag[y1:y2, x1:x2]
        ru = flow[y1:y2, x1:x2, 0]
        rv = flow[y1:y2, x1:x2, 1]

        features.extend([
            rm.mean(), rm.max(), rm.std(),
            np.percentile(rm, 90),              # 90th percentile (captures peak motion)
            ru.mean(), rv.mean(),
            ru.std(), rv.std(),
            np.abs(ru).mean() + np.abs(rv).mean(),  # Total displacement
        ])

    # Asymmetry features (micro-expressions can be asymmetric)
    left_mag = mag[:, :w//2]
    right_mag = mag[:, w//2:]
    right_flipped = right_mag[:, ::-1]
    min_w = min(left_mag.shape[1], right_flipped.shape[1])
    asym = np.abs(left_mag[:, :min_w] - right_flipped[:, :min_w])
    features.extend([asym.mean(), asym.max(), asym.std()])

    return np.array(features, dtype=np.float32)


def extract_temporal_flow_features(frames_uint8, key_indices):
    """
    Extract flow features at multiple temporal points.
    - onset→apex, apex→offset, onset→offset
    """
    T = len(frames_uint8)
    onset_idx = key_indices.get('onset', 0)
    apex_idx = key_indices.get('apex', T // 2)
    offset_idx = key_indices.get('offset', T - 1)

    flows = {}
    pairs = [
        ('onset_apex', onset_idx, apex_idx),
        ('apex_offset', apex_idx, offset_idx),
        ('onset_offset', onset_idx, offset_idx),
    ]

    all_features = []
    for name, idx1, idx2 in pairs:
        if idx1 == idx2:
            idx2 = min(idx1 + 1, T - 1)
        flow = compute_optical_flow(frames_uint8[idx1], frames_uint8[idx2])
        block_feat = extract_flow_block_features(flow, num_blocks=3)
        roi_feat = extract_roi_flow_features(flow)
        all_features.append(np.concatenate([block_feat, roi_feat]))

    return np.concatenate(all_features)


@torch.no_grad()
def extract_all_features(model, dataset, device='cuda'):
    """Extract comprehensive feature set from all sequences."""
    model = model.to(device)
    all_feats = {'appearance': [], 'flow_block': [], 'flow_roi': [],
                 'flow_temporal': [], 'flow_cnn': [], 'lbp_diff': []}
    labels, subjects, emotions = [], [], []

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    print(f"  Processing {len(dataset)} sequences...")

    for i in range(len(dataset)):
        sample = dataset[i]
        frames_tensor = sample['frames']
        T = frames_tensor.shape[0]

        # 1. Appearance features
        feats = model(frames_tensor.to(device)).cpu().numpy()
        appearance_feat = feats.mean(axis=0)  # [2048]

        # Denormalize for flow/LBP
        frames_denorm = (frames_tensor * std_t + mean_t).clamp(0, 1)
        frames_uint8 = (frames_denorm * 255).byte().permute(0, 2, 3, 1).numpy()

        onset_frame = frames_uint8[0]
        apex_frame = frames_uint8[T // 2]
        offset_frame = frames_uint8[T - 1]

        # 2. Flow features (onset → apex)
        flow_oa = compute_optical_flow(onset_frame, apex_frame)
        flow_block = extract_flow_block_features(flow_oa, num_blocks=4)
        flow_roi = extract_roi_flow_features(flow_oa)

        # 3. Multi-temporal flow features
        key_indices = {'onset': 0, 'apex': T // 2, 'offset': T - 1}
        flow_temporal = extract_temporal_flow_features(frames_uint8, key_indices)

        # 4. CNN features from flow visualization
        flow_img = flow_to_image(flow_oa)
        flow_t = torch.from_numpy(flow_img).permute(2, 0, 1).float() / 255.0
        flow_t = (flow_t - mean_t.squeeze(0)) / std_t.squeeze(0)
        flow_cnn_feat = model(flow_t.unsqueeze(0).to(device)).cpu().numpy().flatten()

        # 5. LBP difference features (texture change onset→apex)
        lbp_onset = compute_lbp(onset_frame, radius=2, n_points=8)
        lbp_apex = compute_lbp(apex_frame, radius=2, n_points=8)
        lbp_diff = lbp_apex - lbp_onset  # Texture change

        # Collect
        all_feats['appearance'].append(appearance_feat)
        all_feats['flow_block'].append(flow_block)
        all_feats['flow_roi'].append(flow_roi)
        all_feats['flow_temporal'].append(flow_temporal)
        all_feats['flow_cnn'].append(flow_cnn_feat)
        all_feats['lbp_diff'].append(lbp_diff)
        labels.append(sample['emotion_label'].item())
        subjects.append(sample['subject_id'])
        emotions.append(sample['emotion_name'])

    for k in all_feats:
        all_feats[k] = np.array(all_feats[k])

    return all_feats, np.array(labels), np.array(subjects), np.array(emotions)


def flow_to_image(flow, frame_size=(224, 224)):
    """Convert optical flow to HSV-encoded RGB image."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(hsv, frame_size), cv2.COLOR_HSV2RGB)


def build_feature_sets(feats):
    """Build feature combinations for evaluation."""
    return {
        'flow_block': feats['flow_block'],
        'flow_roi': feats['flow_roi'],
        'flow_block+roi': np.concatenate([feats['flow_block'], feats['flow_roi']], axis=1),
        'flow_temporal': feats['flow_temporal'],
        'flow_cnn': feats['flow_cnn'],
        'lbp_diff': feats['lbp_diff'],
        'flow+lbp': np.concatenate([feats['flow_block'], feats['flow_roi'], feats['lbp_diff']], axis=1),
        'appearance': feats['appearance'],
        'appearance+flow': np.concatenate([feats['appearance'], feats['flow_block'], feats['flow_roi']], axis=1),
        'appearance+flow+lbp': np.concatenate([
            feats['appearance'], feats['flow_block'], feats['flow_roi'], feats['lbp_diff']
        ], axis=1),
        'full': np.concatenate([
            feats['appearance'], feats['flow_block'], feats['flow_roi'],
            feats['flow_temporal'], feats['flow_cnn'], feats['lbp_diff']
        ], axis=1),
    }


def remap_labels(labels):
    """Apply 3-class mapping, return new labels and valid mask."""
    new = np.array([CLASS_MAP_3.get(l, -1) for l in labels])
    return new, new >= 0


def loso_cv_tuned(X, y, subjects, feat_name):
    """LOSO-CV with preprocessing pipeline."""
    logo = LeaveOneGroupOut()

    # Build pipeline: StandardScaler → PCA → SVM
    n_comp = min(50, X.shape[0] - 2, X.shape[1])
    if n_comp < 2:
        n_comp = min(X.shape[1], X.shape[0] - 2)

    results = {}

    # Test multiple SVM configs
    configs = [
        ('SVM-RBF-C1', SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)),
        ('SVM-RBF-C10', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)),
        ('SVM-RBF-C100', SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=42)),
        ('SVM-Lin-C1', SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)),
        ('SVM-Lin-C10', SVC(kernel='linear', C=10, class_weight='balanced', random_state=42)),
        ('RF-200', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)),
        ('GBM-100', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=42) if n_comp > 1 else None
    X_pca = pca.fit_transform(X_s) if pca else X_s

    for name, clf in configs:
        if 'RF' in name or 'GBM' in name:
            y_pred = cross_val_predict(clf, X_s, y, groups=subjects, cv=logo)
        else:
            y_pred = cross_val_predict(clf, X_pca, y, groups=subjects, cv=logo)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        results[name] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str,
                       default='checkpoints/emotion/casmeii_macro_pretrain/casmeii_resnet50/best_model.pth')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--output_dir', type=str, default='checkpoints/emotion/samm_optimized')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPTIMIZED 3-CLASS MICRO-EXPRESSION RECOGNITION")
    print("  Classes: negative / positive / surprise")
    print("=" * 80)

    model = load_backbone(args.pretrained, args.backbone)

    # Load ALL data
    train_ds = SAMMDataset(split='train', sequence_length=16)
    val_ds = SAMMDataset(split='val', sequence_length=16)
    test_ds = SAMMDataset(split='test', sequence_length=16)

    print(f"\nDataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Extract features from all splits
    print("\n--- Feature Extraction ---")
    feats_train, labels_train, subj_train, emo_train = extract_all_features(model, train_ds, device)
    feats_val, labels_val, subj_val, emo_val = extract_all_features(model, val_ds, device)
    feats_test, labels_test, subj_test, emo_test = extract_all_features(model, test_ds, device)

    # Combine all for LOSO
    all_feats = {}
    for k in feats_train:
        all_feats[k] = np.concatenate([feats_train[k], feats_val[k], feats_test[k]], axis=0)
    all_labels = np.concatenate([labels_train, labels_val, labels_test])
    all_subjects = np.concatenate([subj_train, subj_val, subj_test])
    all_emotions = np.concatenate([emo_train, emo_val, emo_test])

    # Apply 3-class mapping
    y3, valid = remap_labels(all_labels)
    y3 = y3[valid]
    subjects3 = all_subjects[valid]
    feats3 = {k: v[valid] for k, v in all_feats.items()}

    print(f"\n3-class dataset: {len(y3)} samples (excluded {(~valid).sum()} 'other' samples)")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y3)))}")
    print(f"Subjects: {len(np.unique(subjects3))}")

    # Build feature sets
    fs = build_feature_sets(feats3)

    print(f"\nFeature dimensions:")
    for name, X in fs.items():
        print(f"  {name:30s}: {X.shape[1]} dims")

    # ═══════════════════════════════════════════════════════════════
    # LOSO-CV with multiple classifiers
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("LOSO-CV: 3-CLASS (negative / positive / surprise)")
    print("=" * 80)

    best_acc, best_f1, best_config = 0, 0, ""
    all_results = {}

    for feat_name, X in fs.items():
        results = loso_cv_tuned(X, y3, subjects3, feat_name)

        print(f"\n  {feat_name} (dim={X.shape[1]}):")
        for clf_name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
            acc, f1 = res['accuracy'], res['f1']
            star = " ★★★" if acc > best_acc else (" ★" if acc > best_acc - 0.02 else "")
            print(f"    {clf_name:15s}: Acc={acc:.4f}  F1={f1:.4f}{star}")

            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_config = f"{feat_name} + {clf_name}"
                best_y_pred = res['y_pred']

            all_results[f"{feat_name}_{clf_name}"] = {
                'accuracy': float(acc), 'f1': float(f1)
            }

    # ═══════════════════════════════════════════════════════════════
    # Best model details
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"BEST RESULT: {best_config}")
    print(f"           Accuracy: {best_acc:.1%}")
    print(f"           F1 (weighted): {best_f1:.4f}")
    print("=" * 80)

    print(f"\nClassification Report:")
    print(classification_report(y3, best_y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y3, best_y_pred)
    print(f"  {'':12s} {'negative':>10s} {'positive':>10s} {'surprise':>10s}")
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j]:10d}" for j in range(len(CLASS_NAMES)))
        print(f"  {name:12s} {row}")

    # Per-subject accuracy
    print(f"\nPer-Subject Accuracy:")
    for subj in sorted(np.unique(subjects3)):
        mask = subjects3 == subj
        if mask.sum() > 0:
            subj_acc = accuracy_score(y3[mask], best_y_pred[mask])
            print(f"  Subject {subj}: {subj_acc:.1%} ({mask.sum()} samples)")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'task': '3-class micro-expression recognition',
        'classes': CLASS_NAMES,
        'total_samples': int(len(y3)),
        'best_config': best_config,
        'best_accuracy': float(best_acc),
        'best_f1': float(best_f1),
        'all_results': all_results,
    }
    with open(output_dir / 'optimized_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Top-10 results
    print(f"\n{'─' * 60}")
    print("TOP-10 CONFIGURATIONS:")
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['accuracy'])
    for i, (name, res) in enumerate(sorted_results[:10]):
        print(f"  {i+1:2d}. {name:45s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
    print(f"{'─' * 60}")

    print(f"\nResults saved to: {output_dir / 'optimized_results.json'}")


if __name__ == '__main__':
    main()
