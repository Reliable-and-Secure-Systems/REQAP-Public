"""
Validate Filter-Level W=A Mixed-Precision Configuration (PTQ)
=============================================================

Applies a per-output-channel (filter) bit-width configuration to the model,
calibrates per-channel activation quantizers, and evaluates accuracy.

Weight quantization:  per-filter, using the list config {layer: [bits_per_ch]}
Activation quantization: per-channel PerChannelActivationQuantizer (same bit list)
W=A constraint:       enforced — weight config == activation config per channel

Usage:
    python validate_config_granular.py \\
        --model levit \\
        --checkpoint models/best3_levit_model_cifar10.pth \\
        --config levit_channel_config_2_4_8_weight.json \\
        --activation-config levit_channel_config_2_4_8_activation.json \\
        --dataset cifar10
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# All Conv variants output channels at dim 1 (B, C, ...).
# Linear variants always output channels at the last dim regardless of rank.
_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from quantization.activations import PerChannelActivationQuantizer
from quantization.bops import compute_bops
from joint_granular_search import get_calib_eval_loaders
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataloader(dataset, batch_size=128, input_size=None, train=False):
    if dataset == 'cifar10':
        return get_cifar10_dataloader(train=train, batch_size=batch_size,
                                      input_size=input_size or 32)
    elif dataset == 'cifar100':
        return get_cifar100_dataloader(train=train, batch_size=batch_size,
                                       input_size=input_size or 32)
    elif dataset == 'gtsrb':
        return get_gtsrb_dataloader(train=train, batch_size=batch_size,
                                    input_size=input_size or 224)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Per-filter weight quantization
# ---------------------------------------------------------------------------

def apply_filter_weight_quantization(model, weight_config):
    """
    Quantize each output filter independently according to weight_config.

    Args:
        model:         PyTorch model
        weight_config: {layer_name: [bits_per_channel]}

    Returns:
        int: number of layers quantized
    """
    count = 0
    for name, module in model.named_modules():
        if name not in weight_config:
            continue
        bits_list = weight_config[name]
        if not isinstance(bits_list, list):
            # Fallback: layer-wise
            q_w, _, _ = quantize_tensor(module.weight.data, bit_width=int(bits_list))
            module.weight.data = q_w
            count += 1
            continue

        w = module.weight.data
        q_w = w.clone()
        for i, bits in enumerate(bits_list):
            if i >= w.shape[0]:
                break
            ch = w[i:i + 1]                                  # (1, ...)
            q_ch, _, _ = quantize_tensor(ch, bit_width=int(bits), method='symmetric')
            q_w[i] = q_ch.squeeze(0)
        module.weight.data = q_w
        count += 1

    return count


# ---------------------------------------------------------------------------
# Per-channel activation quantizer insertion
# ---------------------------------------------------------------------------

def insert_per_channel_quantizers(model, activation_config, device):
    """
    Insert a PerChannelActivationQuantizer after each layer listed in
    activation_config using a forward hook.

    Args:
        model:             PyTorch model
        activation_config: {layer_name: [bits_per_channel]}
        device:            torch device string

    Returns:
        quantizers: list of PerChannelActivationQuantizer instances
    """
    # Clear any pre-existing forward hooks to avoid stacking
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()

    quantizers = []
    bit_dist = {}

    for name, module in model.named_modules():
        if name not in activation_config:
            continue
        bits_list = activation_config[name]
        if not isinstance(bits_list, list):
            # Fallback to scalar — wrap in list of same bit for all channels
            # (shouldn't happen in normal usage)
            num_ch = module.weight.shape[0] if hasattr(module, 'weight') else 1
            bits_list = [int(bits_list)] * num_ch

        # channel_dim is determined from layer type — ground truth, not shape inference.
        # All Conv variants (1d/2d/3d, transposed) output (B, C, ...) → dim 1.
        # All Linear variants output channels at the last dim regardless of rank:
        #   (B,C), (B,N,C) [ViT/LeViT], (B,H,W,C) [Swin], etc.
        if isinstance(module, _CONV_TYPES):
            ch_dim = 1
        elif isinstance(module, nn.Linear):
            ch_dim = -1
        else:
            # Unknown layer type — default to last dim and warn the user.
            ch_dim = -1
            print(
                f'\n[WARNING] Unknown layer type for channel_dim detection:\n'
                f'  Layer : {name}\n'
                f'  Type  : {type(module).__name__}\n'
                f'  Action: Defaulting to channel_dim=-1 (channels at last dim).\n'
                f'  If quantization results look wrong for this layer, identify\n'
                f'  which dimension holds the output channels, then add\n'
                f'  {type(module).__name__} to _CONV_TYPES in validate_config_granular.py\n'
                f'  (if channels are at dim 1) or leave as-is (if at last dim).\n'
            )

        q = PerChannelActivationQuantizer(channel_bits=[int(b) for b in bits_list],
                                          channel_dim=ch_dim)
        q.to(device)
        q.train()   # calibration mode
        quantizers.append(q)

        for b in bits_list:
            bit_dist[b] = bit_dist.get(b, 0) + 1

        def make_hook(quantizer):
            def hook(module, inp, output):
                return quantizer(output)
            return hook

        module.register_forward_hook(make_hook(q))

    total = sum(bit_dist.values())
    print(f'[ACTIVATION QUANT] Inserted {len(quantizers)} per-channel quantizers:')
    for b in sorted(bit_dist.keys(), reverse=True):
        pct = bit_dist[b] / total * 100
        print(f'  A{b}: {bit_dist[b]:6d} channels ({pct:5.1f}%)')

    return quantizers


def calibrate_quantizers(model, quantizers, loader, device, n_batches=10):
    """Run calibration batches to accumulate per-channel running statistics."""
    if not quantizers:
        return
    print(f'[CALIBRATION] Running {n_batches} calibration batches...')
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(images.to(device))
    for q in quantizers:
        q.eval()
    print('[CALIBRATION] Complete.')


# ---------------------------------------------------------------------------
# BOPs calculation (filter-level W=A)
# ---------------------------------------------------------------------------

def compute_bops_filter_level(model_name, dataset, weight_config, activation_config=None,
                               mac_profile_dir=None):
    """
    Compute Bit Operations for filter-level quantization using ground-truth
    per-layer MAC counts from profile_macs.py (torch.utils.flop_counter),
    via the shared quantization.bops.compute_bops(). Replaces the old
    hand-rolled spatial-tracking estimate, which never advanced past
    MaxPool layers and so overstated every CNN's GBOPs — see
    REMEDIATION_PLAN.md Phase A1.

    Args:
        model_name, dataset: used to locate the profile_macs.py output
            (mac_profiles/{model_name}_{dataset}_macs.json).
        weight_config:     {layer: [bits_per_channel]}
        activation_config: {layer: [bits_per_channel]} of W=A layers.
                            Layers absent (Phase-2 rollback) use bw×32.
        mac_profile_dir:   override for the mac_profiles/ directory.

    Returns:
        (baseline_gbops, quantized_gbops)  in GigaBOPs
    """
    if mac_profile_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mac_profile_dir = os.path.join(repo_root, 'mac_profiles')
    profile_path = os.path.join(mac_profile_dir, f'{model_name}_{dataset}_macs.json')
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f'No MAC profile at {profile_path} — run '
            f'`python profile_macs.py --model {model_name} --dataset {dataset}` first '
            f'(see REMEDIATION_PLAN.md Phase A1/B).'
        )
    with open(profile_path) as f:
        profile = json.load(f)

    result = compute_bops(profile['layer_macs'], profile['other_macs'],
                           weight_config, activation_config or {})
    return result['fp32_bops'] / 1e9, result['quantized_bops'] / 1e9


# ---------------------------------------------------------------------------
# W=A constraint verification
# ---------------------------------------------------------------------------

def verify_wa_constraint(weight_config, activation_config):
    """Check that weight and activation configs have identical channel bit lists."""
    mismatches = 0
    for layer in weight_config:
        if layer not in activation_config:
            print(f'  [MISSING] {layer} not in activation config')
            mismatches += 1
            continue
        w_bits = weight_config[layer]
        a_bits = activation_config[layer]
        if isinstance(w_bits, list) and isinstance(a_bits, list):
            if w_bits != a_bits:
                print(f'  [MISMATCH] {layer}: weight {w_bits[:5]}... != activation {a_bits[:5]}...')
                mismatches += 1
        elif w_bits != a_bits:
            print(f'  [MISMATCH] {layer}: weight {w_bits} != activation {a_bits}')
            mismatches += 1
    return mismatches == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate filter-level W=A config (PTQ)')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True,
                        help='Weight config JSON {layer: [bits_per_channel]}')
    parser.add_argument('--activation-config', type=str, required=True,
                        help='Activation config JSON {layer: [bits_per_channel]}')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--calib-batches', type=int, default=64,
                        help='Calibration batches, drawn from a seeded train-split '
                             'subset (default: 64, raised from 10 — see '
                             'REMEDIATION_PLAN.md Phase C addendum)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results-json', type=str, default=None,
                        help='Optional path to save PTQ results as JSON')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Optional path to save the quantized model checkpoint (.pth). '
                             'Saves quantized weights + calibrated activation quantizer states.')
    args = parser.parse_args()

    if args.model == 'swin' and args.batch_size > 16:
        print(f'WARNING: Reducing batch size to 16 for Swin (GPU memory)')
        args.batch_size = 16
    elif args.model == 'levit' and args.batch_size > 32:
        print(f'WARNING: Reducing batch size to 32 for LeViT (GPU memory)')
        args.batch_size = 32

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)
    input_size = 224 if args.model in ('levit', 'swin') or args.dataset == 'gtsrb' else 32

    print('=' * 70)
    print('FILTER-LEVEL W=A PTQ VALIDATION')
    print('=' * 70)

    t0 = time.time()

    # Load configs
    with open(args.config, 'r') as f:
        weight_config = json.load(f)
    with open(args.activation_config, 'r') as f:
        activation_config = json.load(f)

    total_channels = sum(len(v) if isinstance(v, list) else 1
                         for v in weight_config.values())
    all_bits = [b for v in weight_config.values()
                for b in (v if isinstance(v, list) else [v])]
    avg_bits = sum(all_bits) / len(all_bits) if all_bits else 0
    print(f'Weight config:  {len(weight_config)} layers, {total_channels} channels, '
          f'avg {avg_bits:.2f} bits')

    # Verify W=A
    print('\nVerifying W=A constraint...')
    if verify_wa_constraint(weight_config, activation_config):
        print(f'  PASSED: W=A constraint satisfied for all {len(weight_config)} layers')
    else:
        print('  WARNING: W=A constraint violations found above')

    # Measure FP32 baseline accuracy
    print(f'\nMeasuring FP32 baseline...')
    model_fp32 = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model_fp32.to(device)
    loader_baseline = get_dataloader(args.dataset, batch_size=args.batch_size, input_size=input_size)
    baseline_acc = evaluate_accuracy(model_fp32, loader_baseline, device=device)
    del model_fp32
    print(f'  Baseline: {baseline_acc:.2f}%')

    # Load model (FP32)
    t1 = time.time()
    print(f'\nLoading {args.model}...')
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model.to(device)
    print(f'  Model loaded in {time.time()-t1:.2f}s')

    # Apply per-filter weight quantization
    t2 = time.time()
    n_layers = apply_filter_weight_quantization(model, weight_config)
    print(f'Quantized {n_layers} layers per-filter in {time.time()-t2:.2f}s')

    # Insert per-channel activation quantizers
    quantizers = insert_per_channel_quantizers(model, activation_config, device)

    # Load data — calibration uses a SEEDED subset of the TRAIN split
    # (previously reused the test loader, so bit-width decisions upstream
    # were effectively calibrated on the same data used for the reported
    # accuracy; see REMEDIATION_PLAN.md Phase A2 / audit Finding 4).
    # Seeded (not just train=True with unseeded shuffle) so this final
    # validation calibrates on the SAME deterministic subset the search's
    # own quick_ptq_validate/confirmation-eval used — an unseeded random
    # draw here was the actual cause of Swin-T/CIFAR-100 passing its own
    # confirmation eval (9.18% drop) but then measuring 26.12% drop on a
    # freshly, differently, calibrated final run — same config, different
    # random calibration batches, and Swin's activation quantization is
    # documented as unusually sensitive to this. See REMEDIATION_PLAN.md
    # Phase C addendum, 2026-07-20/21. Final accuracy below still uses the
    # unseeded test split, unchanged — only calibration is now seeded.
    loader = get_dataloader(args.dataset, batch_size=args.batch_size, input_size=input_size)
    calib_loader, _ = get_calib_eval_loaders(
        args.dataset, args.batch_size, input_size,
        n_calib_batches=args.calib_batches, n_eval_batches=0)

    # Calibrate
    t3 = time.time()
    calibrate_quantizers(model, quantizers, calib_loader, device, n_batches=args.calib_batches)
    print(f'  Calibration: {time.time()-t3:.2f}s')

    # Evaluate
    t4 = time.time()
    print('\nEvaluating filter-level PTQ accuracy...')
    acc = evaluate_accuracy(model, loader, device=device)
    eval_time = time.time() - t4
    print(f'Filter-Level PTQ Accuracy: {acc:.2f}%')

    # BOPs
    baseline_gbops, quant_gbops = compute_bops_filter_level(
        args.model, args.dataset, weight_config, activation_config=activation_config
    )
    reduction = baseline_gbops / quant_gbops if quant_gbops > 0 else 0

    print('\n' + '=' * 70)
    print('BOPs ANALYSIS (Filter-Level W=A)')
    print('=' * 70)
    print(f'  Baseline (FP32):       {baseline_gbops:>10.2f} GBOPs')
    print(f'  Quantized (Filter-W=A):{quant_gbops:>10.2f} GBOPs')
    print(f'  BOPs Reduction:        {reduction:>10.2f}x')
    print(f'  Average bits (W=A):    {avg_bits:>10.2f}')
    print('=' * 70)

    # Save quantized model checkpoint
    if args.save_model:
        # Collect calibrated quantizer states in layer order.
        # quantizers list is built by iterating model.named_modules() filtered by
        # activation_config, so zipping with that same filtered order reconstructs
        # the name → quantizer mapping without modifying insert_per_channel_quantizers.
        layer_names_ordered = [
            name for name, _ in model.named_modules()
            if name in activation_config
        ]
        quantizer_states = {
            name: {
                'running_min': q.running_min.cpu(),
                'running_max': q.running_max.cpu(),
                'channel_bits': q.channel_bits,
                'channel_dim': q.channel_dim,
                'percentile': q.percentile,
            }
            for name, q in zip(layer_names_ordered, quantizers)
        }
        checkpoint = {
            'model_name': args.model,
            'dataset': args.dataset,
            'model_state_dict': model.state_dict(),
            'quantizer_states': quantizer_states,
            'weight_config': weight_config,
            'activation_config': activation_config,
            'metadata': {
                'avg_bits': round(avg_bits, 3),
                'ptq_accuracy': round(acc, 4),
                'baseline_accuracy': round(baseline_acc, 4),
                'accuracy_drop': round(baseline_acc - acc, 4),
                'bops_reduction': round(reduction, 4),
            },
        }
        torch.save(checkpoint, args.save_model)
        print(f'\n[SAVED] Quantized model checkpoint → {args.save_model}')

    # Save results JSON (used by auto_quantize_engine_granular.py QAT gate)
    if args.results_json:
        results = {
            'ptq_accuracy': round(acc, 4),
            'baseline_accuracy': round(baseline_acc, 4),
            'accuracy_drop': round(baseline_acc - acc, 4),
            'avg_bits': round(avg_bits, 3),
            'baseline_gbops': round(baseline_gbops, 4),
            'quantized_gbops': round(quant_gbops, 4),
            'bops_reduction': round(reduction, 4),
        }
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n[RESULTS] Saved to {args.results_json}')

    total_time = time.time() - t0
    print(f'\nTotal: {total_time:.1f}s ({total_time/60:.1f} min)')


if __name__ == '__main__':
    main()
