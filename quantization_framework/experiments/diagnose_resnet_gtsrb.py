"""
Diagnose which GTSRB evaluation split road_0.9994904891304348.pth actually
matches — its filename encodes ~99.95% accuracy, but the pipeline's default
evaluation (Test.csv) gives only ~43.83%. evaluation/pipeline.py's
get_gtsrb_dataloader() has a documented use_train_val_split=True mode for
"models trained on Train folder split" — this script checks both modes
directly against the real checkpoint to confirm which one it needs, before
touching anything in the real pipeline. See REMEDIATION_PLAN.md Phase C /
SESSION.md 2026-07-20 for context.

Usage:
    python diagnose_resnet_gtsrb.py --checkpoint models/road_0.9994904891304348.pth
"""
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from evaluation.pipeline import get_gtsrb_dataloader, evaluate_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('resnet', checkpoint_path=args.checkpoint, num_classes=43)
    model.to(device)
    model.eval()

    print('\n=== Mode 2 (default): Test.csv ===')
    loader_testcsv = get_gtsrb_dataloader(train=False, batch_size=128, input_size=224)
    acc_testcsv = evaluate_accuracy(model, loader_testcsv, device=device)
    print(f'Accuracy: {acc_testcsv:.2f}%')

    print('\n=== Mode 1: internal Train-folder val split (val_ratio=0.2, seed=42) ===')
    loader_split = get_gtsrb_dataloader(train=False, batch_size=128, input_size=224,
                                        use_train_val_split=True, val_ratio=0.2, seed=42)
    acc_split = evaluate_accuracy(model, loader_split, device=device)
    print(f'Accuracy: {acc_split:.2f}%')

    print('\n=== Verdict ===')
    print(f'Test.csv:            {acc_testcsv:.2f}%')
    print(f'Train-split (val):   {acc_split:.2f}%')
    if acc_split > acc_testcsv + 20:
        print('-> Checkpoint clearly matches the internal Train-folder split.'
              ' Use use_train_val_split=True everywhere for this model.')
    elif acc_testcsv > acc_split + 20:
        print('-> Checkpoint clearly matches Test.csv (the current default).'
              ' The 43.83% result is NOT a split-mismatch bug — something else is wrong.')
    else:
        print('-> Neither mode is dramatically better. Split mismatch is probably'
              ' NOT the explanation — investigate the checkpoint/preprocessing further.')


if __name__ == '__main__':
    main()
