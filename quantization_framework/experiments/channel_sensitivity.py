"""
Per-Filter Sensitivity Analysis (W=A Filter-Level Quantization)
===============================================================

Computes a per-output-channel (filter) sensitivity score for every
Conv2d and Linear layer in the model without running per-filter accuracy sweeps.

Sensitivity metric:
  - Conv2d: weight quantization error (L2 norm of (quantized - FP32) per filter).
            No forward passes needed — runs in seconds.
  - Linear: (weight_error_norm + activation_error_norm) × gradient_norm.
            Weight and activation errors are independently normalized to [0,1]
            across all Linear channels so they contribute on equal footing.
            SUM (not product) ensures a channel is sensitive if EITHER weight
            OR activation quantization hurts.
            Gradient norms collected via backward passes on calibration data.
            Captures actual downstream loss sensitivity for each output channel,
            correctly protecting channels in deep stages (e.g. Swin Stage-3) that
            are close to the classifier head and have large gradient magnitude.

Higher score = more sensitive filter = should receive higher bit-width.

Output CSV columns:  layer, channel_idx, sensitivity  (normalized [0,1] per layer)

Usage:
    python channel_sensitivity.py \\
        --model levit \\
        --checkpoint models/best3_levit_model_cifar10.pth \\
        --dataset cifar10 \\
        --bits 2 4 8

    # Outputs: levit_cifar10_channel_sensitivity.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader,
)
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def get_dataloader(dataset, train=False, batch_size=128, input_size=None):
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
# Weight quantization error per filter
# ---------------------------------------------------------------------------

def compute_weight_error(module, min_bits):
    """
    L2 norm of (quantized - FP32) weight for each output channel.
    No forward passes needed.

    Args:
        module:   Conv2d or Linear module
        min_bits: lowest bit-width candidate (apply quantization at this level)

    Returns:
        list of float, one per output channel (raw L2 error, not normalized)
    """
    weights = module.weight.data
    num_channels = weights.shape[0]
    errors = []

    for c in range(num_channels):
        w_c = weights[c].unsqueeze(0)                              # (1, ...)
        q_c, _, _ = quantize_tensor(w_c, bit_width=min_bits, method='symmetric')
        error = torch.norm(q_c.squeeze(0) - weights[c], p=2).item()
        errors.append(error)

    return errors


# ---------------------------------------------------------------------------
# Gradient norm per output channel (for Linear layers only)
# ---------------------------------------------------------------------------

def collect_gradient_norms(model, linear_layer_names, dataloader, device, n_batches=10,
                           norm_type='rms'):
    """
    Run backward passes and collect per-output-channel gradient RMS norm
    for the specified Linear layers.

    Uses RMS (root-mean-square) instead of L2 norm to remove spatial-size
    dependence.  L2 = sqrt(sum of N squared values) scales with sqrt(N),
    so layers with larger spatial dims (e.g. Swin Stage-0 at 56x56=3136)
    get artificially inflated scores compared to later stages (7x7=49).
    RMS = sqrt(mean of squared values) is independent of N.

    Uses cross-entropy loss on calibration data.  Channels close to the
    classifier head naturally receive large gradient norms regardless of
    architecture, making this metric architecture-agnostic.

    Supports all output layouts (channels always at last dim for Linear):
      (B, C), (B, N, C) [ViT/LeViT], (B, H, W, C) [Swin]

    Returns:
        dict: {layer_name: list of float (one per output channel)}
    """
    accum = {name: None for name in linear_layer_names}
    counts = {name: 0 for name in linear_layer_names}
    hooks = []

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None or g.dim() < 2:
                return
            # channels always at last dim for Linear layers
            # RMS norm: spatial-size invariant (L2 scales with sqrt(N))
            flat = g.detach().reshape(-1, g.shape[-1])
            if norm_type == 'rms':
                norms = flat.pow(2).mean(dim=0).sqrt()   # RMS: spatial-size invariant
            else:
                norms = flat.norm(dim=0)                  # L2: scales with sqrt(N)
            if accum[name] is None:
                accum[name] = norms.cpu()
            else:
                accum[name] += norms.cpu()
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in linear_layer_names:
            h = module.register_full_backward_hook(make_hook(name))
            hooks.append(h)

    # Disable inplace ReLU to avoid conflict with register_full_backward_hook.
    # The hook wraps the Linear output in a custom Function (view); inplace ops
    # on that view corrupt the backward graph. Non-inplace ReLU is identical in value.
    for m in model.modules():
        if isinstance(m, nn.ReLU) and m.inplace:
            m.inplace = False

    criterion = nn.CrossEntropyLoss()
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

    for h in hooks:
        h.remove()
    model.eval()

    result = {}
    for name in linear_layer_names:
        if accum[name] is not None and counts[name] > 0:
            result[name] = (accum[name] / counts[name]).tolist()
        else:
            result[name] = None

    return result


# ---------------------------------------------------------------------------
# Activation L2-norm per channel (kept for reference / ablation)
# ---------------------------------------------------------------------------

def collect_activation_norms(model, linear_layer_names, dataloader, device, n_batches=10):
    """
    Run calibration forward passes and collect per-channel output activation L2-norm
    for the specified Linear layers.

    Supports all output layouts (channels always at last dim for Linear layers):
      (B, C), (B, N, C) [ViT/LeViT], (B, H, W, C) [Swin]

    Returns:
        dict: {layer_name: list of float (one per output channel)}
    """
    accum = {name: None for name in linear_layer_names}
    counts = {name: 0 for name in linear_layer_names}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            # output layouts (channels always at last dim for Linear layers):
            #   (B, C)        — standard Linear
            #   (B, N, C)     — ViT-style Linear (LeViT)
            #   (B, H, W, C)  — Swin-style Linear (channels-last 4D)
            if output.dim() < 2:
                return
            # Flatten all non-channel dims, then norm over the channel dim
            norms = output.detach().reshape(-1, output.shape[-1]).norm(dim=0)  # (C,)
            if accum[name] is None:
                accum[name] = norms.cpu()
            else:
                accum[name] += norms.cpu()
            counts[name] += 1
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if name in linear_layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            images = images.to(device)
            model(images)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average over batches
    result = {}
    for name in linear_layer_names:
        if accum[name] is not None and counts[name] > 0:
            result[name] = (accum[name] / counts[name]).tolist()
        else:
            result[name] = None

    return result


# ---------------------------------------------------------------------------
# Activation quantization error per channel (for Linear layers only)
# ---------------------------------------------------------------------------

def collect_activation_quant_error(model, linear_layer_names, dataloader, device,
                                    min_bits, n_batches=10, percentile=99.99):
    """
    For each Linear layer and each output channel, compute the RMS error when
    the channel's activation is fake-quantized at min_bits.

    Two-phase approach matching PerChannelActivationQuantizer exactly:
      Phase 1 (calibrate): Accumulate running min/max over n_batches with
               momentum=0.9 and percentile clipping (same as validation).
      Phase 2 (measure):   Quantize using calibrated stats, compute RMS error.

    This ensures the error measured here matches the actual quantization error
    during PTQ validation.  The previous single-batch approach used per-batch
    raw min/max, which underestimated error for models like Swin where
    activation distributions are multimodal and outlier-heavy.

    Uses RMS (not L2) for spatial-size invariance — same rationale as Fix 2
    for gradient norms.

    Returns:
        dict: {layer_name: [rms_error_per_channel]}
    """
    # ------- Phase 1: Calibrate running min/max -------
    running_min = {}
    running_max = {}
    initialized = {name: False for name in linear_layer_names}
    momentum = 0.9
    hi = percentile / 100.0      # e.g. 0.9999
    lo = 1.0 - hi                # e.g. 0.0001
    calib_hooks = []

    def make_calib_hook(name):
        def hook(module, input, output):
            if output.dim() < 2:
                return
            C = output.shape[-1]
            flat = output.detach().reshape(-1, C)  # (M, C)

            # Percentile clipping: matches PerChannelActivationQuantizer
            # (prevents outliers from stretching the range)
            ch_min = torch.quantile(flat.float(), lo, dim=0)  # (C,)
            ch_max = torch.quantile(flat.float(), hi, dim=0)  # (C,)

            if not initialized[name]:
                running_min[name] = ch_min.cpu()
                running_max[name] = ch_max.cpu()
                initialized[name] = True
            else:
                running_min[name].mul_(momentum).add_(ch_min.cpu() * (1 - momentum))
                running_max[name].mul_(momentum).add_(ch_max.cpu() * (1 - momentum))
        return hook

    for name, module in model.named_modules():
        if name in linear_layer_names:
            h = module.register_forward_hook(make_calib_hook(name))
            calib_hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            images = images.to(device)
            model(images)

    for h in calib_hooks:
        h.remove()

    # ------- Phase 2: Measure error using calibrated stats -------
    accum = {name: None for name in linear_layer_names}
    counts = {name: 0 for name in linear_layer_names}
    measure_hooks = []

    def make_measure_hook(name):
        def hook(module, input, output):
            if output.dim() < 2:
                return
            if name not in running_min:
                return
            C = output.shape[-1]
            flat = output.detach().reshape(-1, C)  # (M, C)

            # Use calibrated running stats (same as validation quantizer)
            r_min = running_min[name].to(flat.device)  # (C,)
            r_max = running_max[name].to(flat.device)  # (C,)

            q_min = 0
            q_max = 2 ** min_bits - 1
            scale = (r_max - r_min) / q_max
            scale = torch.clamp(scale, min=1e-8)
            zp = (-r_min / scale).round().clamp(q_min, q_max)

            quantized = (flat / scale + zp).round().clamp(q_min, q_max)
            dequantized = (quantized - zp) * scale

            # RMS error per channel: spatial-size invariant
            rms_error = (dequantized - flat).pow(2).mean(dim=0).sqrt()  # (C,)

            if accum[name] is None:
                accum[name] = rms_error.cpu()
            else:
                accum[name] += rms_error.cpu()
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in linear_layer_names:
            h = module.register_forward_hook(make_measure_hook(name))
            measure_hooks.append(h)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            images = images.to(device)
            model(images)

    for h in measure_hooks:
        h.remove()

    result = {}
    for name in linear_layer_names:
        if accum[name] is not None and counts[name] > 0:
            result[name] = (accum[name] / counts[name]).tolist()
        else:
            result[name] = None

    return result


# ---------------------------------------------------------------------------
# Main sensitivity computation
# ---------------------------------------------------------------------------

def compute_channel_sensitivity(model, dataloader, bit_choices, device='cuda',
                                 n_calib_batches=10, gradient_norm='rms'):
    """
    Compute per-filter sensitivity for all Conv2d and Linear layers.

    Args:
        model:          PyTorch model (eval mode, FP32)
        dataloader:     Validation/calibration dataloader
        bit_choices:    List of int bit-widths, e.g. [2, 4, 8]
        device:         Device string
        n_calib_batches: Calibration batches for Linear activation norms

    Returns:
        tuple: (sensitivity, per_bit_act_errors)
          - sensitivity: {layer_name: list of float} — normalized [0, 1] per channel
          - per_bit_act_errors: {layer_name: {"2": [...], "4": [...], ...}} per-bit
            activation quant errors for Linear layers (empty dict for Conv2d-only models)
    """
    min_bits = min(bit_choices)

    # Identify linear layer names (need activation norm collection)
    linear_layer_names = []
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layer_names.append(name)
        elif isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)

    all_layer_names = conv_layer_names + linear_layer_names
    total_layers = len(all_layer_names)
    print(f"\nComputing per-filter sensitivity for {total_layers} layers "
          f"({len(conv_layer_names)} Conv2d, {len(linear_layer_names)} Linear)...")

    # Collect gradient norms for Linear layers (backward passes)
    grad_norms = {}
    if linear_layer_names:
        print(f"[1/3] Collecting gradient norms for {len(linear_layer_names)} "
              f"Linear layers ({n_calib_batches} batches)...")
        grad_norms = collect_gradient_norms(
            model, linear_layer_names, dataloader, device, n_calib_batches,
            norm_type=gradient_norm
        )
    else:
        print("[1/3] No Linear layers — skipping gradient norm collection.")

    # Collect activation quantization error for Linear layers at ALL bit-widths
    # (forward passes). The min_bits errors are used for the sensitivity metric;
    # the full per-bit dict is returned separately for the activation guard post-pass.
    act_quant_errors = {}
    per_bit_act_errors = {}   # {layer: {"2": [...], "4": [...], "8": [...]}}
    if linear_layer_names:
        print(f"[2/3] Collecting activation quantization error for "
              f"{len(linear_layer_names)} Linear layers at {len(bit_choices)} "
              f"bit-widths ({n_calib_batches} calibration + {n_calib_batches} "
              f"measurement batches each)...")
        for bits in sorted(bit_choices):
            print(f"       ... {bits}-bit activations")
            errors_at_bits = collect_activation_quant_error(
                model, linear_layer_names, dataloader, device, bits, n_calib_batches
            )
            for layer_name in linear_layer_names:
                if layer_name not in per_bit_act_errors:
                    per_bit_act_errors[layer_name] = {}
                per_bit_act_errors[layer_name][str(bits)] = errors_at_bits.get(layer_name)
            if bits == min_bits:
                act_quant_errors = errors_at_bits
    else:
        print("[2/3] No Linear layers — skipping activation error collection.")

    # Compute per-filter weight quantization error for all layers
    print(f"[3/3] Computing weight quantization error at {min_bits}-bit...")
    sensitivity = {}

    module_dict = {name: mod for name, mod in model.named_modules()}

    # First pass: compute raw scores for every layer (no normalization yet)
    raw_scores = {}
    for layer_name in all_layer_names:
        module = module_dict[layer_name]
        num_channels = module.weight.shape[0]

        w_errors = compute_weight_error(module, min_bits)

        if layer_name in linear_layer_names:
            # Combined metric: (w_norm + a_norm) × gradient_norm
            # w_norm and a_norm are normalized later (see below).
            # Store raw components for now.
            g_norms = grad_norms.get(layer_name)
            a_errors = act_quant_errors.get(layer_name)
            raw_scores[layer_name] = {
                'w_errors': w_errors,
                'a_errors': a_errors if (a_errors is not None
                                         and len(a_errors) == num_channels) else w_errors,
                'g_norms': g_norms if (g_norms is not None
                                       and len(g_norms) == num_channels) else None,
            }
        else:
            raw_scores[layer_name] = w_errors

    # Normalization strategy:
    #   Conv2d → per-layer [0,1]: scale-invariant, proven stable across CNNs.
    #   Linear → combined metric: (w_norm + a_norm) × gradient_norm
    #     1. Normalize weight errors to [0,1] globally across all Linear channels.
    #     2. Normalize activation errors to [0,1] globally across all Linear channels.
    #     This ensures both components contribute on equal footing regardless of
    #     their absolute magnitude scales.
    #     3. SUM (not product): a channel is sensitive if EITHER weight OR activation
    #        quantization hurts. Product would zero out channels where one component
    #        is near-zero even if the other is high.
    #     4. Multiply by gradient norm: preserves cross-layer importance (deep-stage
    #        channels near the head rank above early-stage channels).
    #     5. Final global [0,1] normalization across all Linear layers.

    # Collect global max for weight and activation errors across all Linear channels
    all_w = [v for ln in linear_layer_names
             for v in (raw_scores[ln]['w_errors'] if isinstance(raw_scores.get(ln), dict) else [])]
    all_a = [v for ln in linear_layer_names
             for v in (raw_scores[ln]['a_errors'] if isinstance(raw_scores.get(ln), dict) else [])]
    w_global_max = max(all_w) if all_w and max(all_w) > 0 else 1.0
    a_global_max = max(all_a) if all_a and max(all_a) > 0 else 1.0

    # Compute combined scores for Linear layers
    combined_linear_scores = {}
    for layer_name in linear_layer_names:
        components = raw_scores[layer_name]
        w_normed = [w / w_global_max for w in components['w_errors']]
        a_normed = [a / a_global_max for a in components['a_errors']]
        g_norms = components['g_norms']

        if g_norms is not None:
            combined = [(w + a) * g for w, a, g in zip(w_normed, a_normed, g_norms)]
        else:
            combined = [w + a for w, a in zip(w_normed, a_normed)]
        combined_linear_scores[layer_name] = combined

    # Global [0,1] normalization across all Linear layers
    all_linear_vals = [v for scores in combined_linear_scores.values() for v in scores]
    linear_global_max = max(all_linear_vals) if all_linear_vals and max(all_linear_vals) > 0 else 1.0

    for layer_name in all_layer_names:
        if layer_name in linear_layer_names:
            sensitivity[layer_name] = [v / linear_global_max
                                       for v in combined_linear_scores[layer_name]]
        else:
            raw = raw_scores[layer_name]
            max_val = max(raw) if max(raw) > 0 else 1.0
            sensitivity[layer_name] = [v / max_val for v in raw]

    return sensitivity, per_bit_act_errors


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_sensitivity_csv(sensitivity, output_path):
    """
    Save sensitivity dict to CSV.

    Columns: layer, channel_idx, sensitivity
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'channel_idx', 'sensitivity'])
        for layer_name, scores in sensitivity.items():
            for c_idx, score in enumerate(scores):
                writer.writerow([layer_name, c_idx, f'{score:.6f}'])
    print(f"\nSaved sensitivity CSV: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Per-filter sensitivity analysis')
    parser.add_argument('--model', type=str, required=True,
                        help='Model: levit, resnet, swin, vgg11_bn')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8],
                        help='Candidate bit-widths (used to determine min_bits)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: <model>_channel_sensitivity.csv)')
    parser.add_argument('--calib-batches', type=int, default=10,
                        help='Calibration batches for Linear activation norms')
    parser.add_argument('--act-error-output', type=str, default=None,
                        help='Output JSON path for per-bit activation errors '
                             '(default: <model>_activation_errors.json)')
    parser.add_argument('--gradient-norm', type=str, default='rms',
                        choices=['rms', 'l2'],
                        help='Gradient norm type for Linear layers: rms (default, '
                             'spatial-size invariant) or l2 (pre-Fix2 ablation)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = args.output or f'{args.model}_{args.dataset}_channel_sensitivity.csv'
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)

    print('=' * 70)
    print('PER-FILTER SENSITIVITY ANALYSIS')
    print('=' * 70)
    print(f'Model:     {args.model}')
    print(f'Dataset:   {args.dataset}')
    print(f'Bits:      {args.bits}  (min={min(args.bits)} used for weight error)')
    print(f'Device:    {device}')
    print('=' * 70)

    t0 = time.time()

    # Load model
    print('\nLoading model...')
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    # Load dataloader
    bs = 32 if args.model in ('levit', 'swin') else 128
    input_size = 224 if args.model in ('levit', 'swin') or args.dataset == 'gtsrb' else 32
    # train=True — sensitivity (gradient norms + activation errors) must be
    # computed on the train split, not test, since bit-width decisions
    # downstream are effectively selected by this signal (see
    # REMEDIATION_PLAN.md Phase A2 / audit Finding 4).
    loader = get_dataloader(args.dataset, train=True, batch_size=bs, input_size=input_size)

    # Compute sensitivity
    sensitivity, per_bit_act_errors = compute_channel_sensitivity(
        model, loader, args.bits, device=device,
        n_calib_batches=args.calib_batches,
        gradient_norm=args.gradient_norm
    )

    # Save per-bit activation errors JSON (for activation guard post-pass)
    if per_bit_act_errors:
        act_err_path = args.act_error_output or f'{args.model}_{args.dataset}_activation_errors.json'
        with open(act_err_path, 'w') as f:
            json.dump(per_bit_act_errors, f, indent=2)
        print(f'\nSaved per-bit activation errors: {act_err_path}')

    # Print summary
    total_channels = sum(len(v) for v in sensitivity.values())
    print(f'\nSummary: {len(sensitivity)} layers, {total_channels} total filters')
    for name, scores in sensitivity.items():
        print(f'  {name:<55} {len(scores):5d} channels  '
              f'avg={sum(scores)/len(scores):.3f}  max={max(scores):.3f}')

    # Save CSV
    save_sensitivity_csv(sensitivity, output_path)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Output: {output_path}')
    print('\nNext step: Run joint_granular_search.py with this CSV')


if __name__ == '__main__':
    main()
