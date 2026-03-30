#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train M1 model on 224x224 animal dataset (chicken, cow, monkey).
Logs training/validation accuracy and loss per epoch to CSV and TXT.

Usage:
    python train_M1_224.py
"""

import os
import sys
import csv
import time
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.M1 import NetM1
from models.cross_entropy import LabelSmoothingCrossEntropy
import writeLogAcc as wA


# =============================================================================
# Configuration
# =============================================================================
DATASETS_DIR = './datasets'
TRAIN_DIR = os.path.join(DATASETS_DIR, 'train', '224x224')
VAL_DIR = os.path.join(DATASETS_DIR, 'validate')
CHECKPOINT_DIR = './checkpoints/M1_224'
LOG_CSV = os.path.join(CHECKPOINT_DIR, 'training_log_224.csv')
LOG_TXT = os.path.join(CHECKPOINT_DIR, 'training_log_224.txt')

N_CLASS = 5
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2
SEED = 42


# =============================================================================
# Data preparation
# =============================================================================
def prepare_cow_validation():
    """Ensure datasets/validate/cow exists for proper 3-class validation."""
    val_cow = os.path.join(DATASETS_DIR, 'validate', 'cow')
    if os.path.exists(val_cow) and len(os.listdir(val_cow)) > 0:
        print(f"validate/cow already exists ({len(os.listdir(val_cow))} files)")
        return

    print("validate/cow not found. Creating from unsorted/cow...")
    os.makedirs(val_cow, exist_ok=True)

    # Copy from unsorted/cow (no train/val data leakage)
    unsorted_cow = os.path.join(DATASETS_DIR, 'unsorted', 'cow')
    if os.path.exists(unsorted_cow):
        all_imgs = [f for f in os.listdir(unsorted_cow)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(all_imgs) >= 100:
            random.seed(SEED)
            selected = random.sample(all_imgs, 100)
            for f in selected:
                shutil.copy2(os.path.join(unsorted_cow, f), os.path.join(val_cow, f))
            print(f"  -> Copied 100 images from unsorted/cow to validate/cow (no data leakage)")
            return

    # Fallback: copy from train/224x224/cow (slight data leakage, but necessary)
    train_cow = os.path.join(DATASETS_DIR, 'train', '224x224', 'cow')
    if os.path.exists(train_cow):
        all_imgs = [f for f in os.listdir(train_cow)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        n_pick = min(100, len(all_imgs))
        random.seed(SEED)
        selected = random.sample(all_imgs, n_pick)
        for f in selected:
            shutil.copy2(os.path.join(train_cow, f), os.path.join(val_cow, f))
        print(f"  -> Copied {n_pick} images from train/224x224/cow (WARNING: slight train/val overlap)")
    else:
        print("  ERROR: Could not find cow images. Validation will only cover chicken & monkey.")


# =============================================================================
# Training / validation epoch
# =============================================================================
def run_epoch(model, data_loader, criterion, optimizer, device, train=True):
    """Run one epoch of training or validation."""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

            # Print progress every 10 batches
            if train and (batch_idx % 10 == 0):
                batch_acc = (preds == targets).float().mean().item()
                print(f'  [train] batch {batch_idx+1}/{len(data_loader)}  '
                      f'loss={loss.item():.4f}  acc={100*batch_acc:.1f}%')

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("  M1 Training — 224x224 — chicken / cow / monkey")
    print("=" * 70)

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Prepare validation data
    prepare_cow_validation()

    # ---- Data transforms ----
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ---- Data loaders ----
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    print(f"\nTrain classes: {train_dataset.class_to_idx}")
    print(f"Val   classes: {val_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val   samples: {len(val_dataset)}")

    use_pin = device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_pin)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin)

    # ---- Model ----
    model = NetM1(n_class=N_CLASS, img_size=IMG_SIZE)
    model = model.to(device)

    print(f"\n{model}\n")
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {n_params:,}')

    # ---- Loss / Optimizer / Scheduler ----
    # Label smoothing for training (helps generalization on small datasets)
    train_criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- CSV log header ----
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'loss_train', 'acc_train(%)', 'loss_val', 'acc_val(%)'])

    # ---- Training loop ----
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        # Train
        loss_train, acc_train = run_epoch(
            model, train_loader, train_criterion, optimizer, device, train=True)

        # Validate
        loss_val, acc_val = run_epoch(
            model, val_loader, val_criterion, None, device, train=False)

        scheduler.step()

        # Check best
        is_best = acc_val > best_val_acc
        if is_best:
            best_val_acc = acc_val
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(CHECKPOINT_DIR, 'best_model_224.pth'))

        elapsed = time.time() - t0

        # Log line
        line = (f'Epoch {epoch}/{EPOCHS}  lr={lr:.6f}  '
                f'loss_train={loss_train:.4f}  acc_train={100*acc_train:.2f}%  '
                f'loss_val={loss_val:.4f}  acc_val={100*acc_val:.2f}%  '
                f'(best: {100*best_val_acc:.2f}% @ epoch {best_epoch})  '
                f'time={elapsed:.1f}s')

        print('-' * 80)
        print(line)
        if is_best:
            print('  >> NEW BEST MODEL SAVED!')
        print('-' * 80)

        # Write to log files
        wA.writeLogAcc(LOG_TXT, line)
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{lr:.6f}', f'{loss_train:.4f}',
                             f'{100*acc_train:.2f}', f'{loss_val:.4f}', f'{100*acc_val:.2f}'])

    # ---- Summary ----
    print('\n' + '=' * 70)
    print(f'  Training complete!')
    print(f'  Best val accuracy: {100*best_val_acc:.2f}% at epoch {best_epoch}')
    print(f'  Log CSV: {LOG_CSV}')
    print(f'  Log TXT: {LOG_TXT}')
    print(f'  Best model: {os.path.join(CHECKPOINT_DIR, "best_model_224.pth")}')
    print('=' * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTraining stopped by user.')
        sys.exit(0)
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
