"""
Ablation study: sensitivity of PTQ accuracy to the depth-aware budget cap constant α.

Cap formula: effective_budget = (α / sqrt(N)) * max_cost
             where N = number of quantized layers.

This script re-runs the greedy search with different α values and validates
each resulting config via quick PTQ, printing a summary table.

Usage:
    python ablation_cap_constant.py \
        --model vgg11_bn \
        --checkpoint /path/to/checkpoint.pt \
        --sensitivity /path/to/channel_sensitivity.csv \
        --dataset cifar10 \
        --target-drop 3.0 \
        --alpha-values 1.0 1.3 1.5 1.70 2.0 2.5 \
        --eval-batches 50

No existing files are modified.
"""

import argparse
import copy
import json
import math
import sys
import os

import torch

# ---------------------------------------------------------------------------
# Import shared utilities from joint_granular_search (read-only imports)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from joint_granular_search import (
    load_sensitivity_csv,
    get_num_channels,
    get_dataloader,
    quick_ptq_validate,
    summarize_config,
)
from joint_granular_search import _CONV_TYPES  # noqa: F401  (needed by quick_ptq_validate)

# Import model loader used inside joint_granular_search
from auto_quantize_engine_granular import load_model  # adjust if your loader is elsewhere


# ---------------------------------------------------------------------------
# Greedy search with configurable α  (mirrors greedy_filter_search exactly,
# but accepts cap_alpha as an explicit argument instead of hardcoding 1.0)
# ---------------------------------------------------------------------------

def greedy_filter_search_with_alpha(sensitivity_dict, model, bit_choices,
                                     target_drop, cap_alpha):
    """
    Identical logic to greedy_filter_search in joint_granular_search.py,
    but uses cap_alpha instead of the hardcoded 1.0.

    cap_alpha=1.0 reproduces the default behaviour exactly.
    """
    sorted_bits = sorted(bit_choices, reverse=True)
    max_bits = sorted_bits[0]
    min_bits = sorted_bits[-1]

    config = {}
    for layer_name in sensitivity_dict:
        num_ch = get_num_channels(model, layer_name)
        if num_ch is None:
            continue
        config[layer_name] = [max_bits] * num_ch

    layer_sizes = {name: len(bits_list) for name, bits_list in config.items()}

    candidates = []
    for layer_name, ch_scores in sensitivity_dict.items():
        if layer_name not in config:
            continue
        for ch_idx, score in ch_scores.items():
            candidates.append((layer_name, ch_idx, score))

    candidates.sort(key=lambda x: x[2])

    total_channels = sum(len(v) for v in config.values())

    # Max possible cost
    max_cost = 0.0
    for layer_name, ch_idx, score in candidates:
        current = max_bits
        while current > min_bits:
            next_b = sorted_bits[sorted_bits.index(current) + 1]
            max_cost += score * (current - next_b) / max_bits / layer_sizes[layer_name]
            current = next_b

    # Depth-aware cap with variable α
    n_layers = len(config)
    depth_cap_frac = cap_alpha / math.sqrt(n_layers)
    budget_cap = depth_cap_frac * max_cost
    effective_budget = min(target_drop, budget_cap)

    print(f'  α={cap_alpha:.2f} | N={n_layers} | cap={depth_cap_frac:.1%} '
          f'| budget={effective_budget:.4f} (max_cost={max_cost:.4f})')

    # Multi-pass greedy
    current_bits_map = {(ln, ci): max_bits for ln, ci, _ in candidates}
    total_cost = 0.0
    moves = []

    changed = True
    while changed:
        changed = False
        for layer_name, ch_idx, score in candidates:
            current = current_bits_map[(layer_name, ch_idx)]
            if current == min_bits:
                continue
            next_b = sorted_bits[sorted_bits.index(current) + 1]
            cost = score * (current - next_b) / max_bits / layer_sizes[layer_name]
            if total_cost + cost > effective_budget:
                continue
            current_bits_map[(layer_name, ch_idx)] = next_b
            total_cost += cost
            moves.append((layer_name, ch_idx, current, next_b, score, cost))
            changed = True

    # Write moves back into config
    for layer_name in config:
        for ch_idx in range(len(config[layer_name])):
            if (layer_name, ch_idx) in current_bits_map:
                config[layer_name][ch_idx] = current_bits_map[(layer_name, ch_idx)]

    return config, moves, total_cost


# ---------------------------------------------------------------------------
# Model / dataset metadata  (mirrors what auto_quantize_engine_granular uses)
# ---------------------------------------------------------------------------

MODEL_META = {
    # model_name: (num_classes, batch_size, input_size)
    'vgg11_bn':    (10,  128, 32),
    'resnet':      (43,  128, 32),
    'levit':       (10,  64,  224),
    'levit100':    (100, 64,  224),
    'swin':        (100, 32,  224),
}


def get_meta(model_name, num_classes, batch_size, input_size):
    """Return (num_classes, batch_size, input_size), CLI args override defaults."""
    defaults = MODEL_META.get(model_name, (num_classes, batch_size, input_size))
    nc  = num_classes  if num_classes  is not None else defaults[0]
    bs  = batch_size   if batch_size   is not None else defaults[1]
    isz = input_size   if input_size   is not None else defaults[2]
    return nc, bs, isz


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def run_ablation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load sensitivity CSV once
    print(f'\nLoading sensitivity: {args.sensitivity}')
    sensitivity_dict = load_sensitivity_csv(args.sensitivity)
    print(f'  Layers: {len(sensitivity_dict)}, '
          f'Channels: {sum(len(v) for v in sensitivity_dict.values())}')

    # Load model once (for channel-count queries only; PTQ uses fresh loads)
    nc, bs, isz = get_meta(args.model, args.num_classes, args.batch_size, args.input_size)
    print(f'\nLoading model: {args.model}  (num_classes={nc})')
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=nc)
    model.eval()

    results = []

    print(f'\n{"="*70}')
    print(f'ABLATION: α values = {args.alpha_values}')
    print(f'{"="*70}')

    for alpha in args.alpha_values:
        print(f'\n--- α = {alpha:.2f} ---')

        # Greedy search
        weight_config, moves, total_cost = greedy_filter_search_with_alpha(
            sensitivity_dict=sensitivity_dict,
            model=model,
            bit_choices=args.bits,
            target_drop=args.target_drop,
            cap_alpha=alpha,
        )

        # Config summary
        _, avg_bits, _ = summarize_config(weight_config)
        print(f'  Avg bits: {avg_bits:.3f} | Total cost used: {total_cost:.4f}')

        # Quick PTQ validation
        print(f'  Running PTQ validation ({args.eval_batches} batches)...')
        acc = quick_ptq_validate(
            model_name=args.model,
            checkpoint=args.checkpoint,
            weight_config=weight_config,
            dataset=args.dataset,
            device=device,
            num_classes=nc,
            batch_size=bs,
            input_size=isz,
            activation_config=weight_config,   # W=A
            n_calib_batches=10,
            n_eval_batches=args.eval_batches,
        )

        results.append({
            'alpha':      alpha,
            'avg_bits':   round(avg_bits, 3),
            'total_cost': round(total_cost, 4),
            'ptq_acc':    round(acc, 2),
        })
        print(f'  PTQ acc: {acc:.2f}%')

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f'\n{"="*70}')
    print(f'ABLATION RESULTS  ({args.model} / {args.dataset})')
    print(f'{"="*70}')
    print(f'{"α":>6}  {"Avg Bits":>10}  {"Cost Used":>10}  {"PTQ Acc (%)":>12}')
    print(f'{"-"*6}  {"-"*10}  {"-"*10}  {"-"*12}')
    for r in results:
        print(f'{r["alpha"]:>6.2f}  {r["avg_bits"]:>10.3f}  '
              f'{r["total_cost"]:>10.4f}  {r["ptq_acc"]:>12.2f}')

    # Save JSON
    out = {
        'model':        args.model,
        'dataset':      args.dataset,
        'target_drop':  args.target_drop,
        'bits':         args.bits,
        'eval_batches': args.eval_batches,
        'results':      results,
    }
    out_path = args.output or f'ablation_alpha_{args.model}_{args.dataset}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ablation: sensitivity of PTQ accuracy to cap constant α')
    parser.add_argument('--model',       type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--sensitivity', type=str, required=True,
                        help='Path to channel_sensitivity.csv')
    parser.add_argument('--dataset',     type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--bits',        type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--target-drop', type=float, default=3.0)
    parser.add_argument('--alpha-values', type=float, nargs='+',
                        default=[1.0, 1.3, 1.5, 1.70, 2.0, 2.5],
                        help='List of α values to test (default: 1.0 1.3 1.5 1.70 2.0 2.5)')
    parser.add_argument('--eval-batches', type=int, default=50,
                        help='Test batches for quick PTQ validation (default: 50)')
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--batch-size',  type=int, default=None)
    parser.add_argument('--input-size',  type=int, default=None)
    parser.add_argument('--output',      type=str, default=None,
                        help='Output JSON path (default: ablation_alpha_<model>_<dataset>.json)')
    args = parser.parse_args()
    run_ablation(args)


if __name__ == '__main__':
    main()
