"""
C-Sweep Experiment: Task Complexity vs. Activation Quantization Tolerance
=========================================================================

Tests the hypothesis that activation quantization tolerance decreases
monotonically with the number of output classes C, independent of network
depth.

Design
------
* Uses a fixed LeViT-384 CIFAR-100 checkpoint and a fixed weight/activation
  config (produced by the main pipeline).
* For each C in {10, 25, 50, 100}: filters the test set to C classes,
  applies the same quantization config, calibrates on filtered data,
  evaluates FP32 baseline and PTQ accuracy.
* Holding the config constant isolates class density as the only variable —
  any change in drop is purely from the decision-margin effect.

Class selection uses a fixed seed so that C=10 ⊆ C=25 ⊆ C=50 ⊆ C=100
(nested subsets), giving a clean monotonic test sequence.

Usage
-----
    python csweep_experiment.py \\
        --checkpoint models/best_levit_model_cifar100.pth \\
        --weight-config levit_cifar100_csweep_weight.json \\
        --activation-config levit_cifar100_csweep_activation.json \\
        --class-counts 10 25 50 100 \\
        --output csweep_results.json
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar100_subset_dataloader,
)
# Reuse quantization helpers from validate_config_granular
from experiments.validate_config_granular import (
    apply_filter_weight_quantization,
    insert_per_channel_quantizers,
    calibrate_quantizers,
)


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------

def select_class_ids(num_classes, total_classes=100, seed=42):
    """
    Select `num_classes` from [0, total_classes) using a fixed seed.

    The selection is nested: ids for C=10 are always the first 10 ids of
    the C=25 selection, which are the first 25 of C=50, etc.
    This means the C=10 test set is a strict subset of the C=100 test set.

    Args:
        num_classes   : int — how many classes to select.
        total_classes : int — pool size (100 for CIFAR-100).
        seed          : int — random seed for reproducibility.

    Returns:
        list of int, sorted, length == num_classes.
    """
    rng = np.random.default_rng(seed)
    all_ids = np.arange(total_classes)
    rng.shuffle(all_ids)
    selected = sorted(all_ids[:num_classes].tolist())
    return selected


# ---------------------------------------------------------------------------
# Single-C evaluation
# ---------------------------------------------------------------------------

def evaluate_one_c(model_name, checkpoint, weight_config, activation_config,
                   class_ids, batch_size, input_size, calib_batches,
                   device, data_path='./data'):
    """
    Evaluate FP32 baseline and PTQ accuracy for a single class subset.

    Loads a fresh model copy for both FP32 and PTQ to avoid stale state.

    Args:
        model_name        : str
        checkpoint        : str — path to checkpoint file
        weight_config     : dict — {layer: [bits_per_channel]}
        activation_config : dict — {layer: [bits_per_channel]}
        class_ids         : list of int — CIFAR-100 class indices to use
        batch_size        : int
        input_size        : int
        calib_batches     : int — calibration batches (matches main pipeline)
        device            : str
        data_path         : str

    Returns:
        dict with keys: num_classes, class_ids, n_test_samples,
                        fp32_accuracy, ptq_accuracy, drop
    """
    C = len(class_ids)
    num_classes_model = 100  # LeViT CIFAR-100 head always has 100 outputs

    eval_loader = get_cifar100_subset_dataloader(
        class_ids, train=False, batch_size=batch_size,
        input_size=input_size, data_path=data_path,
    )
    # Calibrate on the train split, evaluate on the test split — same
    # calib/eval separation as the rest of the pipeline (get_calib_eval_loaders).
    # Previously both loaders used train=False, calibrating and evaluating on
    # the identical held-out subset (train/test leakage).
    calib_loader = get_cifar100_subset_dataloader(
        class_ids, train=True, batch_size=batch_size,
        input_size=input_size, data_path=data_path,
    )

    n_test = len(eval_loader.dataset)
    print(f'\n{"="*60}')
    print(f'  C = {C:3d} classes  |  {n_test} test samples')
    print(f'{"="*60}')

    # ---- FP32 baseline ----
    print(f'  [FP32] Loading model...')
    model_fp32 = load_model(model_name, checkpoint_path=checkpoint,
                            num_classes=num_classes_model)
    model_fp32 = model_fp32.to(device)
    model_fp32.eval()

    t0 = time.time()
    fp32_acc = evaluate_accuracy(model_fp32, eval_loader, device=device)
    print(f'  [FP32] Accuracy: {fp32_acc:.2f}%  ({time.time()-t0:.1f}s)')
    del model_fp32
    torch.cuda.empty_cache()

    # ---- PTQ ----
    print(f'  [PTQ]  Loading model...')
    model_ptq = load_model(model_name, checkpoint_path=checkpoint,
                           num_classes=num_classes_model)
    model_ptq = model_ptq.to(device)

    # Step 1: weight quantization (modifies weights in-place)
    n_w = apply_filter_weight_quantization(model_ptq, weight_config)
    print(f'  [PTQ]  Weight quantization applied to {n_w} layers.')

    # Step 2: insert activation quantizers as forward hooks
    quantizers = insert_per_channel_quantizers(model_ptq, activation_config,
                                               device)

    # Step 3: calibrate on C-class subset (same calib_batches as main pipeline)
    calibrate_quantizers(model_ptq, quantizers, calib_loader, device,
                         n_batches=calib_batches)

    # Step 4: evaluate
    model_ptq.eval()
    t0 = time.time()
    ptq_acc = evaluate_accuracy(model_ptq, eval_loader, device=device)
    drop = fp32_acc - ptq_acc
    print(f'  [PTQ]  Accuracy: {ptq_acc:.2f}%  drop: {drop:.2f}%  '
          f'({time.time()-t0:.1f}s)')
    del model_ptq
    torch.cuda.empty_cache()

    return {
        'num_classes': C,
        'class_ids': class_ids,
        'n_test_samples': n_test,
        'fp32_accuracy': round(fp32_acc, 4),
        'ptq_accuracy': round(ptq_acc, 4),
        'drop': round(drop, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='C-sweep: task complexity vs. activation quantization tolerance'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to LeViT-384 CIFAR-100 checkpoint')
    parser.add_argument('--weight-config', required=True,
                        help='Path to weight config JSON')
    parser.add_argument('--activation-config', required=True,
                        help='Path to activation config JSON')
    parser.add_argument('--class-counts', nargs='+', type=int,
                        default=[10, 25, 50, 100],
                        help='Class counts to sweep (default: 10 25 50 100)')
    parser.add_argument('--calib-batches', type=int, default=64,
                        help='Calibration batches (default: 64)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='DataLoader batch size (default: 32)')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Image input size for LeViT (default: 224)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for class selection (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Torchvision data root (default: ./data)')
    parser.add_argument('--model', type=str, default='levit',
                        help='Model name for load_model() (default: levit)')
    parser.add_argument('--output', type=str, default='csweep_results.json',
                        help='Output JSON path (default: csweep_results.json)')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[WARNING] CUDA not available, falling back to CPU.')
        device = 'cpu'

    # Load configs once
    with open(args.weight_config) as f:
        weight_config = json.load(f)
    with open(args.activation_config) as f:
        activation_config = json.load(f)

    print(f'\nC-Sweep Experiment')
    print(f'  Model      : {args.model}')
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Weight cfg : {args.weight_config}')
    print(f'  Act cfg    : {args.activation_config}')
    print(f'  Class counts: {args.class_counts}')
    print(f'  Seed       : {args.seed}')
    print(f'  Device     : {device}')

    results = []
    for C in sorted(args.class_counts):
        class_ids = select_class_ids(C, total_classes=100, seed=args.seed)
        result = evaluate_one_c(
            model_name=args.model,
            checkpoint=args.checkpoint,
            weight_config=weight_config,
            activation_config=activation_config,
            class_ids=class_ids,
            batch_size=args.batch_size,
            input_size=args.input_size,
            calib_batches=args.calib_batches,
            device=device,
            data_path=args.data_path,
        )
        results.append(result)

    # Save output JSON
    output = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'weight_config': args.weight_config,
        'activation_config': args.activation_config,
        'seed': args.seed,
        'calib_batches': args.calib_batches,
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {args.output}')

    # Print summary table
    print(f'\n{"─"*55}')
    print(f'  {"C":>5}  {"FP32 Acc":>10}  {"PTQ Acc":>10}  {"Drop":>8}')
    print(f'{"─"*55}')
    for r in results:
        flag = '  <-- FAIL (>3%)' if r['drop'] > 3.0 else ''
        print(f'  {r["num_classes"]:>5}  {r["fp32_accuracy"]:>9.2f}%'
              f'  {r["ptq_accuracy"]:>9.2f}%  {r["drop"]:>7.2f}%{flag}')
    print(f'{"─"*55}')

    # Check monotonicity
    drops = [r['drop'] for r in results]
    is_monotonic = all(drops[i] <= drops[i+1] for i in range(len(drops)-1))
    print(f'\nMonotonic drop with C: {"YES — hypothesis supported" if is_monotonic else "NO — see contingency in plan"}')


if __name__ == '__main__':
    main()
