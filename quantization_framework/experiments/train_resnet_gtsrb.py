"""
Train ResNet-18 (this project's models/resnet.py, no-MaxPool architecture)
on GTSRB from scratch. Recovers a checkpoint compatible with the CURRENT
architecture, after best_resnet_model_cifar100_changed.pth was lost and its
substitute (road_0.9994904891304348.pth) turned out to predate the MaxPool
removal (Jan 5, 2026) and is architecturally incompatible. See
REMEDIATION_PLAN.md Phase C / SESSION.md 2026-07-20.

Trains on the Train/ folder (standard augmentation), evaluates each epoch
against a held-out 10% split of Train/ (not Test.csv — Test.csv stays
completely unseen until the final checkpoint is picked, so it remains a
valid held-out test set for the actual paper results), saves the
best-by-held-out-accuracy checkpoint.

Usage:
    python train_resnet_gtsrb.py --epochs 30 --output models/resnet_gtsrb_new.pth
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet import ResNet18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='models/resnet_gtsrb_new.pth')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(args.data_path, 'Train')
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    n_val = int(len(full_dataset) * args.val_ratio)
    n_train = len(full_dataset) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=g)

    # val subset needs the no-augmentation transform, not the train one
    val_dataset_plain = datasets.ImageFolder(train_dir, transform=val_transform)
    val_subset.dataset = val_dataset_plain

    print(f'[data] train={len(train_subset)}, held-out val={len(val_subset)}, '
          f'classes={len(full_dataset.classes)}')

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ResNet18(num_classes=43).to(device)
    print(f'[model] parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_correct, train_total, train_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total

        elapsed = time.time() - t0
        print(f'[epoch {epoch+1}/{args.epochs}] train_acc={train_acc:.2f}% '
              f'val_acc={val_acc:.2f}% loss={train_loss/len(train_loader):.4f} '
              f'[{elapsed:.1f}s]')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f'  -> new best, saved to {args.output}')

    print(f'\n[done] best held-out val accuracy: {best_val_acc:.2f}%')
    print(f'Checkpoint: {args.output}')
    print('Next: evaluate against Test.csv with '
          'quantization_framework/experiments/diagnose_resnet_gtsrb.py '
          '(or just run the Phase C CIFAR job — Test.csv is the default eval split).')


if __name__ == '__main__':
    main()
