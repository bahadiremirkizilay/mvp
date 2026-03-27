"""
Micro-Expression Recognition: Optical Flow + Multi-Feature Pipeline
=====================================================================
Standard micro-expression approach from literature:
    1. Compute dense optical flow (onset → apex) to capture MOTION
    2. Extract hand-crafted + CNN features from flow images
    3. Combine appearance + motion features
    4. Classify with SVM (works well in low-data regimes)

Also supports class reduction (8 → 3/5 classes) for better performance.

Usage:
    python emotion/feature_svm.py --pretrained <path_to_macro_checkpoint>
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV

from emotion.model import EmotionClassifier
from emotion.samm_dataset import SAMMDataset, SAMMConfig


# ─── Class Reduction Mappings ───────────────────────────────────
# 8 classes → 5 classes (common in literature)
CLASS_MAP_5 = {
    0: 0,  # anger → negative
    1: 0,  # contempt → negative
    2: 0,  # disgust → negative
    3: 0,  # fear → negative
    4: 1,  # happiness → positive
    5: 0,  # sadness → negative
    6: 2,  # surprise → surprise
    7: 3,  # neutral/other → other
}
CLASS_NAMES_5 = ['negative', 'positive', 'surprise', 'other']

# 8 classes → 3 classes (excluding "other")
CLASS_MAP_3 = {
    0: 0,  # anger → negative
    1: 0,  # contempt → negative
    2: 0,  # disgust → negative
    3: 0,  # fear → negative
    4: 1,  # happiness → positive
    5: 0,  # sadness → negative
    6: 2,  # surprise → surprise
    7: -1, # neutral/other → EXCLUDE
}
CLASS_NAMES_3 = ['negative', 'positive', 'surprise']


def load_pretrained_backbone(pretrained_path: str, backbone: str = 'resnet50'):
    """Load macro-pretrained ResNet-50 backbone as a feature extractor."""
    model = EmotionClassifier(
        num_classes=8,
        backbone=backbone,
        pretrained=True,
        freeze_backbone=True
    )
    model.classifier = nn.Identity()

    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from: {pretrained_path}")
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
            print(f"  Loaded {len(backbone_state)} backbone tensors")
    else:
        print("  Using ImageNet backbone (no macro pretraining)")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_optical_flow(frame1, frame2):
    """Compute dense optical flow between two frames using Farneback method."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow  # [H, W, 2] (u, v)


def extract_flow_features(flow, num_blocks=4):
    """
    Extract hand-crafted features from optical flow field.

    Features per block:
        - Mean magnitude, max magnitude
        - Mean angle, angular histogram (8 bins)
        - Mean u, mean v, std u, std v
    Total: num_blocks^2 * 14 features
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    features = []
    bh, bw = h // num_blocks, w // num_blocks

    for i in range(num_blocks):
        for j in range(num_blocks):
            block_mag = mag[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_ang = ang[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_u = flow[i*bh:(i+1)*bh, j*bw:(j+1)*bw, 0]
            block_v = flow[i*bh:(i+1)*bh, j*bw:(j+1)*bw, 1]

            # Magnitude stats
            features.append(block_mag.mean())
            features.append(block_mag.max())

            # Angular histogram (8 bins)
            hist, _ = np.histogram(block_ang, bins=8, range=(0, 2*np.pi),
                                    weights=block_mag)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            features.extend(hist.tolist())

            # Displacement stats
            features.extend([block_u.mean(), block_v.mean(),
                           block_u.std(), block_v.std()])

    return np.array(features, dtype=np.float32)


def flow_to_image(flow, frame_size=(224, 224)):
    """Convert optical flow to 3-channel image (HSV encoding) for CNN feature extraction."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2    # Hue = angle
    hsv[..., 1] = 255                       # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value = magnitude

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = cv2.resize(rgb, frame_size)
    return rgb


def extract_all_features(model, dataset, device='cuda'):
    """
    Extract appearance features (CNN backbone) + optical flow features per sequence.

    Returns dict with feature arrays, labels, subjects, emotions.
    """
    model = model.to(device)

    all_appearance_mean = []
    all_appearance_apex = []
    all_flow_handcrafted = []
    all_flow_cnn = []
    labels, subjects, emotions = [], [], []

    print(f"  Processing {len(dataset)} sequences...")

    for i in range(len(dataset)):
        sample = dataset[i]
        frames_tensor = sample['frames']  # [T, C, H, W] normalized
        label = sample['emotion_label'].item()
        subject = sample['subject_id']
        emotion = sample['emotion_name']

        T = frames_tensor.shape[0]

        # ── 1. Appearance features from backbone ──
        with torch.no_grad():
            feats = model(frames_tensor.to(device)).cpu().numpy()  # [T, 2048]
        mean_feat = feats.mean(axis=0)
        apex_feat = feats[T // 2]

        # ── 2. Optical flow features ──
        # Denormalize frames back to uint8 for optical flow
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_denorm = (frames_tensor * std_t + mean_t).clamp(0, 1)
        frames_uint8 = (frames_denorm * 255).byte().permute(0, 2, 3, 1).numpy()  # [T, H, W, C]

        onset_frame = frames_uint8[0]
        apex_frame = frames_uint8[T // 2]

        # Compute onset → apex optical flow
        flow_onset_apex = compute_optical_flow(onset_frame, apex_frame)

        # Hand-crafted flow features (block-based statistics)
        flow_hand = extract_flow_features(flow_onset_apex, num_blocks=4)

        # Also compute temporal flow features (consecutive frames)
        temp_flows = []
        for t in range(0, T - 1, max(1, T // 4)):
            f = compute_optical_flow(frames_uint8[t], frames_uint8[min(t+1, T-1)])
            temp_flows.append(extract_flow_features(f, num_blocks=2))
        if temp_flows:
            temp_flow_feat = np.concatenate(temp_flows)
        else:
            temp_flow_feat = np.zeros(56, dtype=np.float32)

        # CNN features from flow visualization image
        flow_img = flow_to_image(flow_onset_apex)
        flow_tensor = torch.from_numpy(flow_img).permute(2, 0, 1).float() / 255.0
        flow_tensor = (flow_tensor - mean_t.squeeze(0)) / std_t.squeeze(0)
        with torch.no_grad():
            flow_cnn_feat = model(flow_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()

        # ── Collect ──
        all_appearance_mean.append(mean_feat)
        all_appearance_apex.append(apex_feat)
        all_flow_handcrafted.append(np.concatenate([flow_hand, temp_flow_feat]))
        all_flow_cnn.append(flow_cnn_feat)
        labels.append(label)
        subjects.append(subject)
        emotions.append(emotion)

    features = {
        'appearance_mean': np.array(all_appearance_mean),
        'appearance_apex': np.array(all_appearance_apex),
        'flow_hand': np.array(all_flow_handcrafted),
        'flow_cnn': np.array(all_flow_cnn),
    }

    return features, np.array(labels), np.array(subjects), np.array(emotions)


def build_feature_sets(features):
    """Build feature combinations for evaluation."""
    sets = {
        'appearance_mean': features['appearance_mean'],
        'appearance_apex': features['appearance_apex'],
        'flow_hand': features['flow_hand'],
        'flow_cnn': features['flow_cnn'],
        'flow_all': np.concatenate([features['flow_hand'], features['flow_cnn']], axis=1),
        'appearance+flow_hand': np.concatenate([features['appearance_mean'], features['flow_hand']], axis=1),
        'appearance+flow_cnn': np.concatenate([features['appearance_mean'], features['flow_cnn']], axis=1),
        'appearance+flow_all': np.concatenate([
            features['appearance_mean'], features['flow_hand'], features['flow_cnn']
        ], axis=1),
        'full_multimodal': np.concatenate([
            features['appearance_mean'], features['appearance_apex'],
            features['flow_hand'], features['flow_cnn']
        ], axis=1),
    }
    return sets


def remap_labels(labels, class_map):
    """Remap labels according to class_map. Returns new labels and mask of valid samples."""
    new_labels = np.array([class_map.get(l, -1) for l in labels])
    valid = new_labels >= 0
    return new_labels, valid


def evaluate_classifiers(X_train, y_train, X_test, y_test):
    """Evaluate multiple classifiers."""
    results = {}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_components = min(50, X_train_s.shape[0] - 1, X_train_s.shape[1])
    if n_components > 1:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
    else:
        X_train_pca = X_train_s
        X_test_pca = X_test_s

    classifiers = {
        'SVM-RBF': SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', random_state=42),
        'SVM-Linear': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42),
        'RF-200': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=10),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'KNN-5': KNeighborsClassifier(n_neighbors=min(5, len(X_train_pca)), weights='distance'),
    }

    for clf_name, clf in classifiers.items():
        if 'RF' in clf_name or 'GBM' in clf_name:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
        else:
            clf.fit(X_train_pca, y_train)
            y_pred = clf.predict(X_test_pca)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        results[clf_name] = {'accuracy': acc, 'f1_weighted': f1, 'y_pred': y_pred}

    return results


def evaluate_loso_cv(X, y, subjects):
    """LOSO-CV with SVM."""
    logo = LeaveOneGroupOut()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    n_components = min(50, X_s.shape[0] - 2, X_s.shape[1])
    if n_components > 1:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_s)
    else:
        X_pca = X_s

    results = {}
    for name, clf in [
        ('SVM-RBF', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)),
        ('SVM-Linear', SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)),
    ]:
        y_pred = cross_val_predict(clf, X_pca, y, groups=subjects, cv=logo)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        results[name] = {'accuracy': acc, 'f1_weighted': f1, 'y_pred': y_pred}

    return results


def run_evaluation(feature_sets_train, y_train, feature_sets_test, y_test,
                   feature_sets_all, y_all, subjects_all, class_names, title):
    """Run full evaluation pipeline for a given class configuration."""
    print(f"\n{'━' * 80}")
    print(f"  {title}")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}, Classes: {len(class_names)}")
    print(f"  Class distribution (train): {dict(Counter(y_train))}")
    print(f"{'━' * 80}")

    best_acc, best_config = 0, ""
    best_loso_acc, best_loso_config = 0, ""

    # ── Train/Test Split ──
    print("\n  --- Train/Test Split ---")
    for feat_name, X_train in feature_sets_train.items():
        X_test = feature_sets_test[feat_name]
        results = evaluate_classifiers(X_train, y_train, X_test, y_test)

        print(f"\n    {feat_name} (dim={X_train.shape[1]}):")
        for clf, res in results.items():
            acc, f1 = res['accuracy'], res['f1_weighted']
            star = " ★" if acc > best_acc else ""
            print(f"      {clf:15s}: Acc={acc:.4f}  F1={f1:.4f}{star}")
            if acc > best_acc:
                best_acc, best_config = acc, f"{feat_name}+{clf}"

    # ── LOSO-CV ──
    print("\n  --- LOSO-CV ---")
    top_feats = ['flow_hand', 'flow_all', 'appearance+flow_hand', 'appearance+flow_all']
    for feat_name in top_feats:
        if feat_name not in feature_sets_all:
            continue
        X = feature_sets_all[feat_name]
        results = evaluate_loso_cv(X, y_all, subjects_all)

        print(f"\n    {feat_name} (dim={X.shape[1]}):")
        for clf, res in results.items():
            acc, f1 = res['accuracy'], res['f1_weighted']
            star = " ★" if acc > best_loso_acc else ""
            print(f"      {clf:15s}: Acc={acc:.4f}  F1={f1:.4f}{star}")
            if acc > best_loso_acc:
                best_loso_acc, best_loso_config = acc, f"{feat_name}+{clf}"

    print(f"\n  {'─' * 60}")
    print(f"  BEST Train/Test: {best_config} → {best_acc:.1%}")
    print(f"  BEST LOSO:       {best_loso_config} → {best_loso_acc:.1%}")

    return best_acc, best_config, best_loso_acc, best_loso_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str,
                       default='checkpoints/emotion/casmeii_macro_pretrain/casmeii_resnet50/best_model.pth')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--output_dir', type=str, default='checkpoints/emotion/samm_svm_v2')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1-B: Optical Flow + Multi-Feature Pipeline")
    print("=" * 80)

    # Load backbone
    model = load_pretrained_backbone(args.pretrained, args.backbone)

    # Load datasets
    train_ds = SAMMDataset(split='train', sequence_length=16)
    val_ds = SAMMDataset(split='val', sequence_length=16)
    test_ds = SAMMDataset(split='test', sequence_length=16)

    print(f"\nDataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Extract ALL features (appearance + optical flow)
    print("\n--- Extracting TRAIN features ---")
    train_feats, train_labels, train_subj, train_emo = extract_all_features(model, train_ds, device)

    print("--- Extracting VAL features ---")
    val_feats, val_labels, val_subj, val_emo = extract_all_features(model, val_ds, device)

    print("--- Extracting TEST features ---")
    test_feats, test_labels, test_subj, test_emo = extract_all_features(model, test_ds, device)

    # Merge train+val
    merged = {}
    for k in train_feats:
        merged[k] = np.concatenate([train_feats[k], val_feats[k]], axis=0)
    merged_labels = np.concatenate([train_labels, val_labels])
    merged_subj = np.concatenate([train_subj, val_subj])

    # All data for LOSO
    all_feats = {}
    for k in train_feats:
        all_feats[k] = np.concatenate([merged[k], test_feats[k]], axis=0)
    all_labels = np.concatenate([merged_labels, test_labels])
    all_subj = np.concatenate([merged_subj, np.concatenate([test_subj])])
    all_emo = np.concatenate([train_emo, val_emo, test_emo])

    print(f"\nTotal: {len(all_labels)} samples, {len(np.unique(all_subj))} subjects")
    print(f"Class distribution: {dict(sorted(Counter(all_emo).items()))}")

    # ═══════════════════════════════════════════════════════════════
    # Build feature sets
    # ═══════════════════════════════════════════════════════════════
    fs_train = build_feature_sets(merged)
    fs_test = build_feature_sets(test_feats)
    fs_all = build_feature_sets(all_feats)

    results_summary = {}

    # ═══════════════════════════════════════════════════════════════
    # A) 8-class evaluation (original)
    # ═══════════════════════════════════════════════════════════════
    a_acc, a_cfg, a_loso, a_lcfg = run_evaluation(
        fs_train, merged_labels, fs_test, test_labels,
        fs_all, all_labels, all_subj,
        SAMMConfig.EMOTION_LABELS, "A) 8-CLASS EVALUATION"
    )
    results_summary['8_class'] = {
        'train_test_best': f"{a_cfg} → {a_acc:.1%}",
        'loso_best': f"{a_lcfg} → {a_loso:.1%}",
    }

    # ═══════════════════════════════════════════════════════════════
    # B) 5-class evaluation (negative/positive/surprise/other)
    # ═══════════════════════════════════════════════════════════════
    y5_train, _ = remap_labels(merged_labels, CLASS_MAP_5)
    y5_test, _ = remap_labels(test_labels, CLASS_MAP_5)
    y5_all, _ = remap_labels(all_labels, CLASS_MAP_5)
    # All samples valid for 5-class (no exclusion)

    b_acc, b_cfg, b_loso, b_lcfg = run_evaluation(
        fs_train, y5_train, fs_test, y5_test,
        fs_all, y5_all, all_subj,
        CLASS_NAMES_5, "B) 4-CLASS EVALUATION (negative/positive/surprise/other)"
    )
    results_summary['4_class'] = {
        'train_test_best': f"{b_cfg} → {b_acc:.1%}",
        'loso_best': f"{b_lcfg} → {b_loso:.1%}",
    }

    # ═══════════════════════════════════════════════════════════════
    # C) 3-class evaluation (negative/positive/surprise, excl. other)
    # ═══════════════════════════════════════════════════════════════
    y3_train, v_train = remap_labels(merged_labels, CLASS_MAP_3)
    y3_test, v_test = remap_labels(test_labels, CLASS_MAP_3)
    y3_all, v_all = remap_labels(all_labels, CLASS_MAP_3)

    fs3_train = {k: v[v_train] for k, v in fs_train.items()}
    fs3_test = {k: v[v_test] for k, v in fs_test.items()}
    fs3_all = {k: v[v_all] for k, v in fs_all.items()}
    subj3_all = all_subj[v_all]

    c_acc, c_cfg, c_loso, c_lcfg = run_evaluation(
        fs3_train, y3_train[v_train], fs3_test, y3_test[v_test],
        fs3_all, y3_all[v_all], subj3_all,
        CLASS_NAMES_3, "C) 3-CLASS EVALUATION (negative/positive/surprise)"
    )
    results_summary['3_class'] = {
        'train_test_best': f"{c_cfg} → {c_acc:.1%}",
        'loso_best': f"{c_lcfg} → {c_loso:.1%}",
    }

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for cfg, res in results_summary.items():
        print(f"\n  {cfg}:")
        print(f"    Train/Test: {res['train_test_best']}")
        print(f"    LOSO-CV:    {res['loso_best']}")

    # Save
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pretrained': args.pretrained,
        'results': results_summary,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
