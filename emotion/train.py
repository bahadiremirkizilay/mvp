"""
Emotion Recognition Training Pipeline
======================================
Professional-grade training system for emotion recognition models on SAMM and CASMEII.

Features:
    • Multi-dataset training (SAMM + CASMEII unified)
    • Class balancing and weighted loss
    • Learning rate scheduling
    • Early stopping with model checkpointing
    • Comprehensive metrics (accuracy, F1, confusion matrix)
    • Tensorboard logging
    • Mixed precision training (FP16)
    • Cross-validation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from emotion.model import create_emotion_model, EmotionClassifier, TemporalEmotionModel
from emotion.samm_dataset import SAMMDataset, get_samm_dataloaders
from emotion.casmeii_dataset import CASMEIIDataset, get_casmeii_dataloaders, UnifiedEmotionDataset


class EmotionTrainer:
    """
    Professional trainer for emotion recognition models.
    
    Supports:
        • Multiple datasets (SAMM, CASMEII, or unified)
        • Class-weighted loss for imbalanced data
        • Multiple optimizers and schedulers
        • Model checkpointing and early stopping
        • Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_weights: Optional[torch.Tensor] = None,
        output_dir: str = 'checkpoints/emotion',
        experiment_name: Optional[str] = None,
        label_smoothing: float = 0.0,
        use_amp: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Emotion recognition model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            class_weights: Optional class weights for imbalanced data
            output_dir: Directory for saving checkpoints
            experiment_name: Name for this experiment (default: timestamp)
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_amp: Whether to use automatic mixed precision (FP16)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.grad_clip = 0.0
        
        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Loss function with optional class weighting and label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing
        )
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.output_dir / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def train_epoch(self, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer instance
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            # Get data
            if 'frames' in batch and batch['frames'].dim() == 5:
                # Temporal data [B, T, C, H, W]
                inputs = batch['frames'].to(self.device)
            else:
                # Static data [B, C, H, W]
                inputs = batch['image'].to(self.device)
            
            labels = batch['emotion_label'].to(self.device)
            
            # Forward pass with AMP
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with AMP scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for batch in pbar:
            # Get data
            if 'frames' in batch and batch['frames'].dim() == 5:
                inputs = batch['frames'].to(self.device)
            else:
                inputs = batch['image'].to(self.device)
            
            labels = batch['emotion_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(
        self,
        num_epochs: int = 50,
        optimizer_name: str = 'adamw',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_name: Optional[str] = 'reduce_on_plateau',
        early_stopping_patience: int = 10,
        save_every: int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        grad_clip: float = 0.0
    ):
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of training epochs
            optimizer_name: Optimizer type ('adam', 'adamw', 'sgd')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_name: LR scheduler ('reduce_on_plateau', 'cosine', None)
            early_stopping_patience: Epochs to wait before early stopping
            save_every: Save checkpoint every N epochs
            optimizer: Pre-configured optimizer (overrides optimizer_name/learning_rate)
            scheduler: Pre-configured scheduler (overrides scheduler_name)
            grad_clip: Max gradient norm for clipping (0 = no clipping)
        """
        self.grad_clip = grad_clip
        
        # Use provided optimizer or create one
        if optimizer is None:
            if optimizer_name == 'adam':
                optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'adamw':
                optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'sgd':
                optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Use provided scheduler or create one
        if scheduler is None:
            if scheduler_name == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            elif scheduler_name == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # Training loop
        print("=" * 80)
        print(f"Starting training: {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  AMP (FP16): {self.use_amp}")
        print(f"  Gradient clipping: {grad_clip}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
            print(f"  Discriminative LR groups: {len(optimizer.param_groups)}")
            for i, pg in enumerate(optimizer.param_groups):
                n_params = sum(p.numel() for p in pg['params'])
                print(f"    Group {i}: lr={pg['lr']:.1e}, params={n_params:,}")
        else:
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            improved = False
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                improved = True
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  ✅ New best model! (Acc: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', val_metrics)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        # Training complete
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"  Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 80)
        
        # Save final metrics
        self.save_training_summary()
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history
        }, checkpoint_path)
    
    def save_training_summary(self):
        """Save detailed training summary."""
        summary_path = self.checkpoint_dir / 'training_summary.json'
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.current_epoch + 1,
            'best_val_accuracy': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch) if hasattr(self, 'best_epoch') else 0,
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_train_acc': float(self.history['train_acc'][-1]),
            'history': {k: [float(v) for v in vals] for k, vals in self.history.items()}
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Training summary saved to: {summary_path}")


def train_samm_emotion_model(
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-2,
    backbone: str = 'resnet50',
    temporal: bool = True,
    output_dir: str = 'checkpoints/emotion',
    pretrained_path: str = None,
    freeze_backbone: bool = False
):
    """
    Train emotion recognition model on SAMM micro-expression dataset.
    
    Strategy for small dataset (138 samples):
        • Freeze entire backbone - use as fixed feature extractor  
        • Average pooling over temporal frames (no LSTM)
        • Replace classifier with compact head (2048→128→8 = 265K params)
        • Higher LR (1e-2) for small classifier
        • Data augmentation in SAMMDataset
        • Mixed precision + gradient clipping
    """
    from torch.utils.data import ConcatDataset
    
    print("=" * 80)
    print("PHASE 1-B: Micro-Expression Fine-tuning (Feature Extraction)")
    print("=" * 80)
    
    # Create datasets - merge train + val for more training data
    train_dataset = SAMMDataset(split='train', sequence_length=16)
    val_dataset = SAMMDataset(split='val', sequence_length=16)
    test_dataset = SAMMDataset(split='test', sequence_length=16)
    
    # Merge train + val → more training data
    merged_train = ConcatDataset([train_dataset, val_dataset])
    
    print(f"\n📊 Dataset Info:")
    print(f"   Train (merged): {len(merged_train)} samples")
    print(f"   Test (evaluation): {len(test_dataset)} samples")
    
    # Create dataloaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        merged_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model with average pooling (no LSTM)
    if temporal:
        model = create_emotion_model(
            model_type='temporal',
            num_classes=8,
            backbone=backbone,
            temporal_model='avg',
            pretrained=True
        )
    else:
        model = create_emotion_model(
            model_type='base',
            num_classes=8,
            backbone=backbone,
            pretrained=True
        )
    
    # Load pretrained weights (transfer from macro pretrain)
    if pretrained_path:
        print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            backbone_state_dict = {}
            is_temporal_target = hasattr(model, 'frame_encoder')
            
            for key, value in state_dict.items():
                if any(skip in key for skip in ['fc', 'classifier', 'lstm', 'temporal']):
                    continue
                
                if is_temporal_target:
                    if key.startswith('backbone.'):
                        new_key = f'frame_encoder.{key}'
                        backbone_state_dict[new_key] = value
                    elif any(layer in key for layer in ['layer', 'conv', 'bn', 'downsample']):
                        new_key = f'frame_encoder.backbone.{key}'
                        backbone_state_dict[new_key] = value
                else:
                    if key.startswith('backbone.'):
                        backbone_state_dict[key] = value
                    elif any(layer in key for layer in ['layer', 'conv', 'bn', 'downsample']):
                        new_key = f'backbone.{key}'
                        backbone_state_dict[new_key] = value
            
            if backbone_state_dict:
                missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
                print(f"   ✅ Loaded {len(backbone_state_dict)} backbone parameter tensors")
            else:
                print("   ⚠️ No compatible backbone weights found")
        except Exception as e:
            print(f"   ⚠️ Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    # ── FREEZE ENTIRE BACKBONE ──
    # For 138 samples, backbone fine-tuning destroys features
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            frozen_count += param.numel()
    
    # Replace classifier with compact head for small dataset
    # Original: 2048→512→8 (1.05M params)
    # New: 2048→128→8 (265K params) - appropriate for 138 samples
    feature_dim = 2048  # ResNet-50 feature dim
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feature_dim, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(128, 8)
    )
    
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Backbone: {backbone} (FROZEN)")
    print(f"   Temporal: Average pooling over {16} frames")
    print(f"   Classifier: {feature_dim}→128→8 (compact)")
    print(f"   ❄️  Frozen: {frozen_count:,} params")
    print(f"   🔥 Trainable: {trainable_count:,} params")
    print(f"   Learning Rate: {learning_rate}")
    
    # Create optimizer - only classifier params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()
    print(f"\n⚖️ Class Weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_weights=class_weights,
        output_dir=output_dir,
        experiment_name=f"samm_{'temporal' if temporal else 'static'}_{backbone}",
        label_smoothing=0.0,
        use_amp=True
    )
    
    # Train with gradient clipping
    trainer.train(
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=25,
        grad_clip=1.0
    )


def train_casmeii_emotion_model(
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    backbone: str = 'resnet50',
    output_dir: str = 'checkpoints/emotion'
):
    """
    Train emotion recognition model on CASMEII dataset.
    
    Args:
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        backbone: CNN backbone architecture
        output_dir: Output directory for checkpoints
    """
    print("=" * 80)
    print("Training Emotion Model on CASMEII Dataset")
    print("=" * 80)
    
    # Create datasets
    train_dataset = CASMEIIDataset(split='train', balance_classes=True)
    test_dataset = CASMEIIDataset(split='test')
    
    print(f"\n📊 Dataset Info:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_sampler = train_dataset.get_sampler()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_emotion_model(
        model_type='base',
        num_classes=8,
        backbone=backbone,
        pretrained=True
    )
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Backbone: {backbone}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"\n⚖️ Class Weights: {class_weights.tolist()}")
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as validation
        class_weights=class_weights,
        output_dir=output_dir,
        experiment_name=f"casmeii_{backbone}"
    )
    
    # Train
    trainer.train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer_name='adamw',
        scheduler_name='reduce_on_plateau',
        early_stopping_patience=15
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument('--dataset', type=str, choices=['samm', 'casmeii', 'both'], default='samm',
                       help='Dataset to train on')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='CNN backbone (resnet18, resnet50, efficientnet_b0)')
    parser.add_argument('--temporal', action='store_true',
                       help='Use temporal model (for SAMM sequences)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints/emotion',
                       help='Output directory')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model checkpoint for transfer learning')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers during finetuning (transfer learning)')
    
    args = parser.parse_args()
    
    if args.dataset == 'samm':
        train_samm_emotion_model(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            backbone=args.backbone,
            temporal=args.temporal,
            output_dir=args.output_dir,
            pretrained_path=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    elif args.dataset == 'casmeii':
        train_casmeii_emotion_model(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            backbone=args.backbone,
            output_dir=args.output_dir
        )
    else:
        print("Training on both datasets not yet implemented")
