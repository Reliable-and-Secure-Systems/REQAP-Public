"""
Joint Greedy Search: Filter-Level W=A Co-Optimized Bit-Width Assignment
=======================================================================

Reads the per-filter sensitivity CSV produced by channel_sensitivity.py and
assigns per-output-channel bit-widths using a greedy reduction algorithm.

W=A constraint: weight channel i and output activation channel i always receive
the same bit-width. The same list is written to both the weight and activation
config files.

Algorithm:
    1. Start all filters at max_bits.
    2. Sort all (layer, channel_idx) pairs globally by sensitivity (ascending).
       Least sensitive channels are reduced first.
    3. Greedily reduce each channel to the next lower bit tier.
       Cost of each reduction: sensitivity × (bits_curr - bits_next) / max_bits
       Budget: total_cost ≤ min(target_drop, depth_aware_cap)
       The depth-aware cap scales as 1/sqrt(N) where N is the number of
       quantized layers, accounting for noise compounding through deep models.
    4. Unsampled channels (skipped due to subsampling in channel_sensitivity.py)
       are filled at max_bits (conservative default).
    5. Output: per-layer list of bit-widths, same for weight and activation.

Usage:
    python joint_granular_search.py \\
        --model levit \\
        --checkpoint models/best3_levit_model_cifar10.pth \\
        --sensitivity levit_channel_sensitivity.csv \\
        --dataset cifar10 \\
        --bits 2 4 8 \\
        --target-drop 3.0
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader,
)
from quantization.primitives import quantize_tensor
from quantization.activations import PerChannelActivationQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sensitivity_csv(csv_path):
    """
    Load per-filter sensitivity CSV (layer, channel_idx, sensitivity).

    Returns:
        dict: {layer_name: {channel_idx: float}}
    """
    sensitivity = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            idx = int(row['channel_idx'])
            score = float(row['sensitivity'])
            if layer not in sensitivity:
                sensitivity[layer] = {}
            sensitivity[layer][idx] = score
    return sensitivity


def get_num_channels(model, layer_name):
    """Return weight.shape[0] (number of output channels/filters) for a layer."""
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            return module.weight.shape[0]
    return None


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


def get_calib_eval_loaders(dataset, batch_size, input_size, n_calib_batches,
                            n_eval_batches, seed=42):
    """
    Disjoint, seeded calibration and rollback-eval loaders, both drawn from
    the TRAIN split. Previously calibration and rollback decisions used the
    test set, and calibration/eval batches within quick_ptq_validate came
    from the same unshuffled iterator (overlapping — eval reused the exact
    calibration batches). See REMEDIATION_PLAN.md Phase A2 / audit Finding 4.
    """
    base_loader = get_dataloader(dataset, train=True, batch_size=batch_size,
                                 input_size=input_size)
    base_dataset = base_loader.dataset
    n = len(base_dataset)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    calib_size = min(n_calib_batches * batch_size, n)
    eval_size = min(n_eval_batches * batch_size, n - calib_size)
    calib_idx = perm[:calib_size]
    eval_idx = perm[calib_size:calib_size + eval_size]

    calib_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(base_dataset, calib_idx), batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(base_dataset, eval_idx), batch_size=batch_size, shuffle=False)
    return calib_loader, eval_loader


# ---------------------------------------------------------------------------
# Greedy filter-level W=A search
# ---------------------------------------------------------------------------

def greedy_filter_search(sensitivity_dict, model, bit_choices, target_drop):
    """
    Assign per-filter bit-widths using a sensitivity-guided greedy algorithm.

    Cost formula (per channel reduction):
        cost = sensitivity × (bits_curr - bits_next) / max_bits / n_channels_in_layer

    Normalising by n_channels_in_layer means the total cost for reducing an
    entire layer equals avg_sensitivity x bit_reduction / max_bits — the same
    scale as the layer-wise sensitivity scores.

    Args:
        sensitivity_dict: {layer: {channel_idx: float}} normalized [0,1] scores.
        model:            PyTorch model (used only to get num_channels per layer).
        bit_choices:      List of int bit-widths, e.g. [2, 4, 8].
        target_drop:      Cumulative sensitivity budget (same scale as layer-wise
                          --target-drop). Higher = more aggressive compression.

    Returns:
        config:      {layer_name: [bits_per_channel]}  (full list for each layer)
        moves:       Log of greedy moves.
        total_cost:  Cumulative cost of reductions made.
    """
    sorted_bits = sorted(bit_choices, reverse=True)   # e.g. [8, 4, 2]
    max_bits = sorted_bits[0]
    min_bits = sorted_bits[-1]

    # Build full config: start all channels at max_bits
    # Unsampled indices (subsampling in channel_sensitivity.py) default to max_bits
    config = {}
    for layer_name in sensitivity_dict:
        num_ch = get_num_channels(model, layer_name)
        if num_ch is None:
            continue
        config[layer_name] = [max_bits] * num_ch

    # Pre-compute layer sizes for cost normalisation
    layer_sizes = {name: len(bits_list) for name, bits_list in config.items()}

    # Flatten to list of (layer, channel_idx, sensitivity_score)
    candidates = []
    for layer_name, ch_scores in sensitivity_dict.items():
        if layer_name not in config:
            continue
        for ch_idx, score in ch_scores.items():
            candidates.append((layer_name, ch_idx, score))

    # Sort ascending: least sensitive first → reduce these first
    candidates.sort(key=lambda x: x[2])

    total_channels = sum(len(v) for v in config.values())

    # Compute max possible cost (reducing everything to min_bits)
    max_cost = 0.0
    for layer_name, ch_idx, score in candidates:
        current = max_bits
        while current > min_bits:
            next_b = sorted_bits[sorted_bits.index(current) + 1]
            max_cost += score * (current - next_b) / max_bits / layer_sizes[layer_name]
            current = next_b

    # ---- Depth-aware budget cap ----
    #
    # The additive cost model (total_cost = sum of per-channel costs) assumes
    # that quantizing channels independently causes damage that sums linearly.
    # In practice, quantization noise COMPOUNDS through sequential layers:
    # each layer processes the noisy output of the previous one, amplifying
    # errors.  The compounding effect grows roughly as sqrt(N) for N sequential
    # quantized layers (random-walk model: independent per-layer noise adds
    # in quadrature).  Residual connections partially decorrelate the noise,
    # making sqrt(N) a reasonable middle ground between fully independent
    # (sqrt(N)) and fully correlated (linear N) models.
    #
    # α=1.0 gives the theoretically pure form: cap(N) = 1/sqrt(N) × C_max.
    # No empirical multiplier — the 1/sqrt(N) scaling alone implements the
    # random-walk noise model.  Example caps at α=1.0:
    #
    #   N=11 (VGG)    → 30%
    #   N=21 (ResNet) → 22%
    #   N=63 (LeViT)  → 13%
    #   N=53 (Swin)   → 14%
    #
    # This is architecture-agnostic: it only depends on the number of quantized
    # layers, not on model-specific knowledge.  Shallow models get generous
    # compression budgets; deep models get conservative ones.
    n_layers = len(config)
    base_factor = 1.0   # α=1.0: cap = 1/sqrt(N) × C_max, the direct expression
                        # of the random-walk noise model with no empirical scaling.
    depth_cap_frac = base_factor / math.sqrt(n_layers)
    budget_cap = depth_cap_frac * max_cost
    effective_budget = min(target_drop, budget_cap)
    budget_capped = effective_budget < target_drop

    print(f'\nTotal layers:   {n_layers}')
    print(f'Total channels: {total_channels}')
    print(f'Profiled:       {len(candidates)} channels')
    print(f'Bit choices:    {sorted_bits}')
    print(f'Max capacity:   {max_cost:.4f}')
    print(f'Requested:      {target_drop:.2f}')
    print(f'Depth cap:      {depth_cap_frac:.1%} of capacity  '
          f'(sqrt(8/{n_layers}) scaling for {n_layers} quantized layers)')
    if budget_capped:
        print(f'Budget CAPPED:  {effective_budget:.4f}  '
              f'({depth_cap_frac:.1%} of max capacity {max_cost:.4f})')
    else:
        print(f'Budget:         {effective_budget:.4f}  '
              f'(within {depth_cap_frac:.1%} cap of {budget_cap:.4f})')
    print()
    print('=' * 70)
    print('GREEDY FILTER-LEVEL REDUCTION')
    print('=' * 70)

    total_cost = 0.0
    moves = []

    # Multi-pass greedy: repeat until no more reductions fit within the budget.
    # Each pass allows channels already reduced (e.g. 8→4) to be reduced again
    # (4→2) in a subsequent pass, enabling multi-tier assignments.
    pass_num = 0
    while True:
        pass_num += 1
        made_progress = False

        for layer_name, ch_idx, score in candidates:
            current_bits = config[layer_name][ch_idx]
            if current_bits == min_bits:
                continue
            try:
                next_bits = sorted_bits[sorted_bits.index(current_bits) + 1]
            except (IndexError, ValueError):
                continue

            # Cost normalised by layer size so budget is scale-independent:
            # total cost for reducing a full layer = avg_sensitivity x bit_reduction / max_bits
            n_ch = layer_sizes[layer_name]
            cost = score * (current_bits - next_bits) / max_bits / n_ch

            if total_cost + cost > effective_budget:
                continue

            config[layer_name][ch_idx] = next_bits
            total_cost += cost
            made_progress = True
            moves.append({
                'layer': layer_name,
                'channel': ch_idx,
                'from': current_bits,
                'to': next_bits,
                'sensitivity': round(score, 4),
                'cost': round(cost, 6),
                'total_cost': round(total_cost, 6),
            })

        if not made_progress:
            break  # Budget exhausted or all channels at min_bits

    print(f'  (completed {pass_num} greedy pass(es))')
    return config, moves, total_cost


# ---------------------------------------------------------------------------
# Activation guard post-pass
# ---------------------------------------------------------------------------

def activation_guard_postpass(config, act_errors, bit_choices, model, threshold_k=3.0):
    """
    For each Linear channel, check activation quant error at assigned bit-width.
    If error > threshold_k * median (across all Linear channels at their assigned
    bits), bump W=A to next higher bit-width.

    Args:
        config:      {layer: [bits_per_channel]} from greedy search
        act_errors:  {layer: {"2": [...], "4": [...], "8": [...]}} per-bit errors
        bit_choices: sorted desc e.g. [8, 4, 2]
        model:       PyTorch model (to identify Linear layers)
        threshold_k: multiplier on median error to trigger bump (default 3.0)

    Returns:
        config (modified in place), num_bumped
    """
    sorted_bits = sorted(bit_choices, reverse=True)  # e.g. [8, 4, 2]
    max_bits = sorted_bits[0]

    # Identify Linear layers present in both config and act_errors
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in config and name in act_errors:
            linear_layers.add(name)

    if not linear_layers:
        print('\n[ACT GUARD] No Linear layers with activation error data — skipping.')
        return config, 0

    # Step 1: Collect activation error at each channel's assigned bit-width
    assigned_errors = []  # list of (layer, ch_idx, error, assigned_bits)
    for layer_name in linear_layers:
        layer_act = act_errors[layer_name]
        bits_list = config[layer_name]
        for ch_idx, assigned_bits in enumerate(bits_list):
            bits_key = str(assigned_bits)
            if bits_key in layer_act and layer_act[bits_key] is not None:
                err = layer_act[bits_key][ch_idx]
                assigned_errors.append((layer_name, ch_idx, err, assigned_bits))

    if not assigned_errors:
        print('\n[ACT GUARD] No activation error data found — skipping.')
        return config, 0

    # Step 2: Compute median and threshold
    all_errors = [e[2] for e in assigned_errors]
    median_err = float(np.median(all_errors))
    threshold = threshold_k * median_err

    print(f'\n{"=" * 70}')
    print('ACTIVATION GUARD POST-PASS')
    print(f'{"=" * 70}')
    print(f'  Linear channels checked: {len(assigned_errors)}')
    print(f'  Median act error:        {median_err:.6f}')
    print(f'  Threshold (k={threshold_k}):     {threshold:.6f}')

    # Step 3: Bump channels above threshold
    num_bumped = 0
    bump_details = {}  # {layer: count}
    for layer_name, ch_idx, err, assigned_bits in assigned_errors:
        if err > threshold and assigned_bits != max_bits:
            # Find next higher bit-width
            curr_idx = sorted_bits.index(assigned_bits)
            if curr_idx > 0:
                new_bits = sorted_bits[curr_idx - 1]
                config[layer_name][ch_idx] = new_bits
                num_bumped += 1
                bump_details[layer_name] = bump_details.get(layer_name, 0) + 1

    # Summary
    if num_bumped > 0:
        print(f'  Channels bumped:         {num_bumped}')
        print(f'  Per-layer bumps:')
        for layer_name in sorted(bump_details.keys()):
            total_ch = len(config[layer_name])
            print(f'    {layer_name:<50} {bump_details[layer_name]:4d}/{total_ch}')
    else:
        print(f'  No channels exceeded threshold — config unchanged.')

    # New bit distribution
    bit_counts, avg_bits, total_ch = summarize_config(config)
    print(f'  Avg bits after guard:    {avg_bits:.2f}')
    print(f'{"=" * 70}')

    return config, num_bumped


# ---------------------------------------------------------------------------
# Quick PTQ validation (for accuracy-aware rollback)
# ---------------------------------------------------------------------------

# Conv types for channel_dim detection (same as validate_config_granular.py)
_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)


def quick_ptq_validate(model_name, checkpoint, weight_config, dataset, device,
                       num_classes, batch_size, input_size,
                       activation_config=None,
                       n_calib_batches=64, n_eval_batches=64):
    """
    Fast PTQ validation using a subset of the test set.

    Loads a fresh model, applies per-filter weight quantization, inserts
    per-channel activation quantizers, calibrates, and evaluates on
    n_eval_batches.  Returns estimated accuracy.

    Args:
        weight_config:     {layer: [bits_per_channel]} for weight quantization.
        activation_config: {layer: [bits_per_channel]} for activation quantization.
                           If None, uses weight_config (W=A for all layers).
                           Layers absent from activation_config get weight-only
                           quantization (no activation hook).
        batch_size:  Batch size for dataloader (caller decides, no model-specific logic).
        input_size:  Input spatial size (caller decides, no model-specific logic).
    """
    if activation_config is None:
        activation_config = weight_config

    # Load fresh model (avoid contamination from previous quantization)
    model = load_model(model_name, checkpoint_path=checkpoint, num_classes=num_classes)
    model.to(device)
    model.eval()

    # Apply per-filter weight quantization (same as validate_config_granular.py)
    for name, module in model.named_modules():
        if name not in weight_config:
            continue
        bits_list = weight_config[name]
        if not isinstance(bits_list, list):
            q_w, _, _ = quantize_tensor(module.weight.data, bit_width=int(bits_list))
            module.weight.data = q_w
            continue
        w = module.weight.data
        q_w = w.clone()
        for i, bits in enumerate(bits_list):
            if i >= w.shape[0]:
                break
            ch = w[i:i + 1]
            q_ch, _, _ = quantize_tensor(ch, bit_width=int(bits), method='symmetric')
            q_w[i] = q_ch.squeeze(0)
        module.weight.data = q_w

    # Insert per-channel activation quantizers (only for layers in activation_config)
    quantizers = []
    for name, module in model.named_modules():
        if name not in activation_config:
            continue
        bits_list = activation_config[name]
        if not isinstance(bits_list, list):
            num_ch = module.weight.shape[0] if hasattr(module, 'weight') else 1
            bits_list = [int(bits_list)] * num_ch

        if isinstance(module, _CONV_TYPES):
            ch_dim = 1
        else:
            ch_dim = -1

        q = PerChannelActivationQuantizer(
            channel_bits=[int(b) for b in bits_list], channel_dim=ch_dim)
        q.to(device)
        q.train()  # calibration mode
        quantizers.append(q)

        def make_hook(quantizer):
            def hook(mod, inp, output):
                return quantizer(output)
            return hook
        module.register_forward_hook(make_hook(q))

    # Disjoint, seeded train-split loaders — calibration and rollback-eval
    # no longer share batches or touch the test set (see
    # get_calib_eval_loaders / REMEDIATION_PLAN.md Phase A2).
    calib_loader, eval_loader = get_calib_eval_loaders(
        dataset, batch_size, input_size, n_calib_batches, n_eval_batches)

    # Calibrate
    with torch.no_grad():
        for images, _ in calib_loader:
            model(images.to(device))
    for q in quantizers:
        q.eval()

    # Evaluate on the disjoint holdout
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    del model
    torch.cuda.empty_cache()

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


# ---------------------------------------------------------------------------
# Accuracy-aware rollback (binary search over layers)
# ---------------------------------------------------------------------------

def accuracy_aware_rollback(config, model_name, checkpoint, dataset, device,
                            num_classes, baseline_acc, max_drop, bit_choices,
                            batch_size, input_size,
                            act_errors=None, n_eval_batches=64, margin=2.0):
    """
    Validate PTQ accuracy after greedy search.  If accuracy drops more than
    max_drop, progressively bump layers back to max_bits using binary search.

    Strategy:
      1. Quick PTQ validation of current config (~2 min).
      2. If within threshold → done.
      3. Sort layers by activation damage score (layers that cause most harm first).
      4. Binary search: bump top-K most damaging layers to max_bits.
         ~6 iterations × ~2 min = ~12 min total.

    Args:
        config:       {layer: [bits_per_channel]} from greedy search
        model_name:   model identifier for loading
        checkpoint:   checkpoint path
        dataset:      dataset name
        device:       torch device string
        num_classes:  number of output classes
        baseline_acc: FP32 baseline accuracy
        max_drop:     maximum allowed accuracy drop (%)
        bit_choices:  list of bit-widths e.g. [2, 4, 8]
        batch_size:   batch size for dataloader
        input_size:   input spatial size
        act_errors:   optional {layer: {"2": [...], "4": [...]}} for scoring layers
        n_eval_batches: batches for quick validation (default 64 — raised from 20;
            at batch_size~32-128 the old 20-batch estimate had a standard error
            on the same order as the delta=3% threshold it was testing against,
            see REMEDIATION_PLAN.md Phase A3 / audit Finding 6)
        margin: safety margin (percentage points) subtracted from max_drop when
            selecting the Phase-2 K — avoids picking a knife-edge K that barely
            passes on this eval but fails under different calibration sampling
            (observed: Swin-T/CIFAR-100 confirmed at 9.18% drop on one calibration
            draw, then measured 26.12% drop on an independently-recalibrated final
            validation — same config. See REMEDIATION_PLAN.md Phase C addendum,
            2026-07-20/21). Falls back to the unmargined max_drop if nothing
            passes with the margin applied, so a tight budget never has literally
            no candidate.

    Returns:
        (weight_config, activation_config, n_layers_affected)
        weight_config and activation_config may differ if Phase 2 triggers:
        layers removed from activation_config get weight-only quantization.
    """
    sorted_bits = sorted(bit_choices, reverse=True)
    max_bits = sorted_bits[0]

    print(f'\n{"=" * 70}')
    print('ACCURACY-AWARE VALIDATION GATE')
    print(f'{"=" * 70}')
    print(f'  Max allowed drop: {max_drop:.1f}%')
    print(f'  Baseline:         {baseline_acc:.2f}%')
    print(f'  Eval batches:     {n_eval_batches}')

    # Step 1: Quick validate current config
    t0 = time.time()
    acc = quick_ptq_validate(model_name, checkpoint, config, dataset, device,
                             num_classes, batch_size, input_size,
                             n_eval_batches=n_eval_batches)
    drop = baseline_acc - acc
    elapsed = time.time() - t0
    print(f'  Quick PTQ:        {acc:.2f}% (drop {drop:.2f}%)  [{elapsed:.1f}s]')

    if drop <= max_drop:
        print(f'  PASSED — within {max_drop:.1f}% threshold, no rollback needed.')
        print(f'{"=" * 70}')
        return config, config, 0

    print(f'  FAILED — drop {drop:.2f}% > {max_drop:.1f}% threshold.')
    print(f'  Starting binary search over layers...')

    # Step 2: Score layers by activation damage
    # Layers with more channels at low bits AND higher activation error get bumped first
    reduced_layers = []
    for layer_name, bits_list in config.items():
        n_reduced = sum(1 for b in bits_list if b < max_bits)
        if n_reduced == 0:
            continue

        # Damage score: combines how many channels were reduced with how much
        # activation error they have at their assigned bit-width
        if act_errors and layer_name in act_errors:
            layer_act = act_errors[layer_name]
            total_err = 0.0
            for ch_idx, assigned_bits in enumerate(bits_list):
                if assigned_bits < max_bits:
                    bits_key = str(assigned_bits)
                    if bits_key in layer_act and layer_act[bits_key] is not None:
                        total_err += layer_act[bits_key][ch_idx]
            damage_score = total_err
        else:
            # Fallback: more reduced channels = more risky
            damage_score = float(n_reduced)

        reduced_layers.append((layer_name, n_reduced, damage_score))

    # Sort by damage score descending (most damaging layers first)
    reduced_layers.sort(key=lambda x: x[2], reverse=True)

    print(f'  Layers with reduced channels: {len(reduced_layers)}')
    print(f'  Top-5 most damaging:')
    for name, n_red, score in reduced_layers[:5]:
        total_ch = len(config[name])
        print(f'    {name:<50} {n_red:4d}/{total_ch} reduced  score={score:.4f}')

    # Step 3: Binary search — find minimum number of layers to bump to max_bits
    # to bring accuracy within threshold
    lo, hi = 0, len(reduced_layers)
    best_k = len(reduced_layers)  # worst case: bump all layers
    best_acc = acc                # start with current (pre-rollback) accuracy

    while lo <= hi:
        mid = (lo + hi) // 2

        # Build test config: bump top-mid layers to max_bits
        test_config = {l: list(v) for l, v in config.items()}
        for i in range(mid):
            layer_name = reduced_layers[i][0]
            test_config[layer_name] = [max_bits] * len(test_config[layer_name])

        t1 = time.time()
        acc = quick_ptq_validate(model_name, checkpoint, test_config, dataset,
                                 device, num_classes, batch_size, input_size,
                                 n_eval_batches=n_eval_batches)
        drop = baseline_acc - acc
        elapsed = time.time() - t1

        bit_counts, avg_bits, _ = summarize_config(test_config)
        print(f'  [K={mid:3d}/{len(reduced_layers)}] '
              f'acc={acc:.2f}% drop={drop:.2f}% avg_bits={avg_bits:.2f}  [{elapsed:.1f}s]')

        if drop <= max_drop:
            best_k = mid
            best_acc = acc
            hi = mid - 1   # try bumping fewer layers
        else:
            # Track best accuracy seen (for accurate reporting)
            if acc > best_acc:
                best_acc = acc
            lo = mid + 1   # need to bump more layers

    # Apply the best Phase 1 config
    n_bumped = 0
    for i in range(best_k):
        layer_name = reduced_layers[i][0]
        old_bits = config[layer_name]
        config[layer_name] = [max_bits] * len(old_bits)
        n_bumped += sum(1 for b in old_bits if b < max_bits)

    phase1_drop = baseline_acc - best_acc
    bit_counts, avg_bits, total_ch = summarize_config(config)
    print(f'\n  Phase 1 complete (bump to {max_bits}-bit):')
    print(f'    Layers bumped: {best_k}/{len(reduced_layers)}')
    print(f'    Channels bumped:  {n_bumped}')
    print(f'    Best accuracy:    {best_acc:.2f}% (drop {phase1_drop:.2f}%)')
    print(f'    Avg bits (W=A):   {avg_bits:.2f}')

    # Check if Phase 1 succeeded
    if phase1_drop <= max_drop:
        print(f'  Phase 1 PASSED — within {max_drop:.1f}% threshold.')
        print(f'  Bit distribution:')
        for b in sorted(bit_counts.keys(), reverse=True):
            count = bit_counts[b]
            pct = count / total_ch * 100
            print(f'    W{b}/A{b}: {count:5d} channels ({pct:5.1f}%)')
        print(f'{"=" * 70}')
        return config, config, best_k

    # ------------------------------------------------------------------
    # Phase 2: Disable activation quantization per-layer
    # ------------------------------------------------------------------
    # Phase 1 bumped all layers to max_bits but accuracy is still too low.
    # This means activation quantization at max_bits is itself harmful for
    # some layers (e.g. attention layers in transformers where quantization
    # noise is amplified by softmax).
    #
    # Strategy: binary search to find the minimum set of layers that need
    # activation quantization DISABLED (weight-only) to meet the threshold.
    # Weight config stays unchanged — only the activation config shrinks.
    print(f'\n  Phase 1 FAILED — drop {phase1_drop:.2f}% even at all-{max_bits}-bit.')
    print(f'  Phase 2: disabling activation quantization per-layer...')

    # Score layers by activation damage for Phase 2.
    # Use act_errors if available; otherwise fall back to channel count.
    phase2_layers = []
    for layer_name, bits_list in config.items():
        if act_errors and layer_name in act_errors:
            # Sum activation error at max_bits for this layer
            max_key = str(max_bits)
            layer_act = act_errors[layer_name]
            if max_key in layer_act and layer_act[max_key] is not None:
                total_err = sum(layer_act[max_key])
            else:
                total_err = float(len(bits_list))
        else:
            total_err = float(len(bits_list))
        phase2_layers.append((layer_name, total_err))

    # Sort by damage descending (most damaging activation quantization first)
    phase2_layers.sort(key=lambda x: x[1], reverse=True)

    print(f'  Layers to search: {len(phase2_layers)}')
    print(f'  Top-5 most damaging (act error at {max_bits}-bit):')
    for name, score in phase2_layers[:5]:
        n_ch = len(config[name])
        print(f'    {name:<50} {n_ch:4d} ch  score={score:.4f}')

    lo2, hi2 = 0, len(phase2_layers)
    best_k2 = len(phase2_layers)
    best_acc2 = best_acc  # start from Phase 1's best
    evaluated2 = []  # (k, acc, drop) for every K tested — binary search's

    while lo2 <= hi2:
        mid2 = (lo2 + hi2) // 2

        # Build activation config: remove top-mid2 layers (weight-only for them)
        test_act_config = dict(config)  # start with full W=A
        for i in range(mid2):
            layer_name = phase2_layers[i][0]
            if layer_name in test_act_config:
                del test_act_config[layer_name]

        t2 = time.time()
        acc2 = quick_ptq_validate(model_name, checkpoint, config, dataset,
                                  device, num_classes, batch_size, input_size,
                                  activation_config=test_act_config,
                                  n_eval_batches=n_eval_batches)
        drop2 = baseline_acc - acc2
        elapsed2 = time.time() - t2
        evaluated2.append((mid2, acc2, drop2))

        n_act_layers = len(test_act_config)
        n_wonly = mid2
        print(f'  [P2 K={mid2:3d}/{len(phase2_layers)}] '
              f'acc={acc2:.2f}% drop={drop2:.2f}% '
              f'act_layers={n_act_layers} w_only={n_wonly}  [{elapsed2:.1f}s]')

        if drop2 <= max_drop:
            best_k2 = mid2
            best_acc2 = acc2
            hi2 = mid2 - 1
        else:
            if acc2 > best_acc2:
                best_acc2 = acc2
            lo2 = mid2 + 1

    # Binary search assumes accuracy improves monotonically as more layers
    # fall back to weight-only (larger K) — not guaranteed in practice, so
    # the traversal's own last-found boundary isn't trustworthy. The
    # objective is to MINIMIZE K (fewest weight-only layers = most
    # compression) subject to the accuracy constraint — so among
    # everything actually evaluated, pick the smallest K that passes
    # (not the most accurate one, which ignores how loose max_drop is and
    # makes the search insensitive to the budget — see REMEDIATION_PLAN.md
    # Phase C, discovered when delta=3/10/15% budget-sweep results came
    # back identical). The confirmation-eval step below (added in the
    # same fix as this one) is what actually guards against the noise
    # that caused the original bug — this selection only needs to pick
    # the right K, not the safest one.
    passing2 = [(k, acc, drop) for k, acc, drop in evaluated2 if drop <= max_drop - margin]
    if not passing2:
        # nothing clears the margin — fall back to the plain (unmargined) budget
        # rather than leaving no candidate at all
        passing2 = [(k, acc, drop) for k, acc, drop in evaluated2 if drop <= max_drop]
    if passing2:
        best_k2, best_acc2, _ = min(passing2, key=lambda x: x[0])

    # Build final activation config
    activation_config = dict(config)
    n_disabled = 0
    disabled_layers = []
    for i in range(best_k2):
        layer_name = phase2_layers[i][0]
        if layer_name in activation_config:
            del activation_config[layer_name]
            n_disabled += 1
            disabled_layers.append(layer_name)

    # Confirmation eval: the winning K's accuracy above was measured once
    # during the binary search traversal (a single noisy sample) — re-verify
    # with a fresh eval before accepting, stepping K up if it fails. See
    # REMEDIATION_PLAN.md Phase A3 / audit Finding 6.
    while True:
        confirm_acc = quick_ptq_validate(model_name, checkpoint, config, dataset,
                                         device, num_classes, batch_size, input_size,
                                         activation_config=activation_config,
                                         n_eval_batches=n_eval_batches)
        confirm_drop = baseline_acc - confirm_acc
        print(f'  [P2 CONFIRM K={best_k2:3d}/{len(phase2_layers)}] '
              f'acc={confirm_acc:.2f}% drop={confirm_drop:.2f}%')
        if confirm_drop <= max_drop or best_k2 >= len(phase2_layers):
            best_acc2 = confirm_acc
            break
        layer_name = phase2_layers[best_k2][0]
        best_k2 += 1
        if layer_name in activation_config:
            del activation_config[layer_name]
            n_disabled += 1
            disabled_layers.append(layer_name)

    phase2_drop = baseline_acc - best_acc2
    print(f'\n  Phase 2 complete (disable activation quantization):')
    print(f'    Layers with act quant disabled: {n_disabled}/{len(phase2_layers)}')
    if disabled_layers:
        print(f'    Disabled layers:')
        for name in disabled_layers:
            n_ch = len(config[name])
            print(f'      {name} ({n_ch} ch) -> weight-only')
    print(f'    Final accuracy:   {best_acc2:.2f}% (drop {phase2_drop:.2f}%)')
    n_act_ch = sum(len(v) for v in activation_config.values())
    n_wonly_ch = total_ch - n_act_ch
    print(f'    W+A channels:     {n_act_ch}')
    print(f'    Weight-only ch:   {n_wonly_ch}')

    print(f'  Bit distribution (weight config):')
    for b in sorted(bit_counts.keys(), reverse=True):
        count = bit_counts[b]
        pct = count / total_ch * 100
        print(f'    W{b}: {count:5d} channels ({pct:5.1f}%)')
    print(f'{"=" * 70}')

    return config, activation_config, best_k + n_disabled


# ---------------------------------------------------------------------------
# Config summary
# ---------------------------------------------------------------------------

def summarize_config(config):
    bit_counts = {}
    total_ch = 0
    total_bits = 0
    for bits_list in config.values():
        for b in bits_list:
            bit_counts[b] = bit_counts.get(b, 0) + 1
            total_ch += 1
            total_bits += b
    avg_bits = total_bits / total_ch if total_ch > 0 else 0
    return bit_counts, avg_bits, total_ch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Filter-level W=A greedy search')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sensitivity', type=str, required=True,
                        help='Path to channel_sensitivity.csv')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Sensitivity budget (same scale as layer-wise --target-drop)')
    parser.add_argument('--output', type=str, default=None,
                        help='Base name for output JSON files')
    parser.add_argument('--act-errors', type=str, default=None,
                        help='Path to per-bit activation errors JSON '
                             '(enables activation guard post-pass for Linear layers)')
    parser.add_argument('--act-threshold', type=float, default=3.0,
                        help='Activation guard threshold multiplier on median error '
                             '(default: 3.0)')
    parser.add_argument('--max-drop', type=float, default=None,
                        help='Max allowed PTQ accuracy drop %%. Enables accuracy-aware '
                             'rollback: after greedy search, validate PTQ accuracy and '
                             'bump layers back to max_bits if drop exceeds this. '
                             'Uses binary search (~12 min for Swin). '
                             'Disabled by default.')
    parser.add_argument('--eval-batches', type=int, default=64,
                        help='Number of test batches for quick PTQ validation '
                             'during accuracy-aware rollback (default: 64, raised '
                             'from 20 — see REMEDIATION_PLAN.md Phase A3)')
    parser.add_argument('--margin', type=float, default=2.0,
                        help='Safety margin (pp) subtracted from --max-drop when '
                             'selecting the Phase-2 K, to avoid knife-edge configs '
                             'that pass calibration noise instead of the real budget '
                             '(default: 2.0, see REMEDIATION_PLAN.md Phase C addendum)')
    parser.add_argument('--weight-only', action='store_true',
                        help='Disable activation quantization entirely. '
                             'Weight config is assigned normally via greedy search; '
                             'activation config is written as empty (no hooks inserted '
                             'during validation). Use for architectures where activation '
                             'quantization is not feasible (e.g. deep transformers).')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bits_str = '_'.join(map(str, sorted(args.bits)))
    base_name = args.output or f'{args.model}_channel_config_{bits_str}'
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)

    print('=' * 70)
    print('JOINT FILTER-LEVEL W=A GREEDY SEARCH')
    print('=' * 70)
    print(f'Model:         {args.model}')
    print(f'Sensitivity:   {args.sensitivity}')
    print(f'Bits:          {sorted(args.bits)}')
    print(f'Budget:        {args.target_drop}')
    print(f'Constraint:    W=A (weight bits = activation bits per channel)')
    print('=' * 70)

    t0 = time.time()

    # Load model (for num_channels lookup only)
    print('\nLoading model...')
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model.eval()

    # Load sensitivity
    print(f'Loading sensitivity profile: {args.sensitivity}')
    sensitivity = load_sensitivity_csv(args.sensitivity)
    print(f'Loaded {len(sensitivity)} layers')

    # Measure baseline accuracy
    bs = 32 if args.model in ('levit', 'swin') else 128
    input_size = 224 if args.model in ('levit', 'swin') or args.dataset == 'gtsrb' else 32
    loader = get_dataloader(args.dataset, train=False, batch_size=bs, input_size=input_size)
    device_t = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device_t)
    print('\nMeasuring FP32 baseline accuracy...')
    baseline_acc = evaluate_accuracy(model, loader, device=device_t)
    print(f'Baseline (test set, for final reporting): {baseline_acc:.2f}%')

    # Separate baseline measured on the SAME train-holdout eval_loader that
    # accuracy_aware_rollback's quick_ptq_validate calls use — comparing a
    # holdout-measured PTQ accuracy against a test-set baseline would bias
    # every "drop" computed during rollback (different data, different
    # sample count). See REMEDIATION_PLAN.md Phase A2.
    _, holdout_eval_loader = get_calib_eval_loaders(
        args.dataset, bs, input_size, n_calib_batches=10, n_eval_batches=args.eval_batches)
    baseline_acc_holdout = evaluate_accuracy(model, holdout_eval_loader, device=device_t)
    print(f'Baseline (train holdout, for rollback decisions): {baseline_acc_holdout:.2f}%')

    # Run greedy search
    config, moves, total_cost = greedy_filter_search(
        sensitivity, model, args.bits, args.target_drop
    )

    # Weight-only mode: skip all activation quantization steps
    if args.weight_only:
        print('\n[WEIGHT-ONLY MODE] Activation quantization disabled.')
        print('  Weight config: assigned by greedy search (mixed-precision)')
        print('  Activation config: empty (no activation quantizers inserted)')
        activation_config = {}
    else:
        # Activation guard post-pass (Linear layers only)
        if args.act_errors:
            with open(args.act_errors, 'r') as f:
                act_errors = json.load(f)
            config, n_bumped = activation_guard_postpass(
                config, act_errors, sorted(args.bits, reverse=True), model,
                args.act_threshold
            )

    # Accuracy-aware rollback: validate PTQ accuracy, bump layers if needed
    # Returns separate weight and activation configs (may differ if Phase 2 triggers)
    if not args.weight_only:
        activation_config = {l: list(v) for l, v in config.items()}
    if not args.weight_only and args.max_drop is not None:
        act_err_data = None
        if args.act_errors:
            with open(args.act_errors, 'r') as f:
                act_err_data = json.load(f)
        config, activation_config, n_layers_bumped = accuracy_aware_rollback(
            config, args.model, args.checkpoint, args.dataset, device_t,
            num_classes, baseline_acc_holdout, args.max_drop,
            sorted(args.bits, reverse=True),
            batch_size=bs, input_size=input_size,
            act_errors=act_err_data,
            n_eval_batches=args.eval_batches,
            margin=args.margin
        )

    weight_config = config

    # Summary
    w_counts, w_avg, w_total = summarize_config(weight_config)
    a_counts, a_avg, a_total = summarize_config(activation_config)
    n_act_layers = len(activation_config)
    n_wonly_layers = len(weight_config) - n_act_layers

    print('\n' + '=' * 70)
    print('SEARCH COMPLETE')
    print('=' * 70)
    print(f'Moves made:      {len(moves)}')
    print(f'Total cost:      {total_cost:.4f}')
    print(f'Total channels:  {w_total}')
    print(f'Avg weight bits: {w_avg:.2f}')
    if n_wonly_layers > 0:
        print(f'W+A layers:      {n_act_layers}  ({a_total} channels)')
        print(f'Weight-only:     {n_wonly_layers} layers '
              f'({w_total - a_total} channels, act quant disabled)')
    else:
        print(f'Avg bits (W=A):  {w_avg:.2f}')
    print(f'\nWeight bit distribution:')
    for b in sorted(w_counts.keys(), reverse=True):
        count = w_counts[b]
        pct = count / w_total * 100
        print(f'  W{b}: {count:5d} channels ({pct:5.1f}%)')
    if n_wonly_layers > 0 and a_total > 0:
        print(f'Activation bit distribution ({n_act_layers} layers):')
        for b in sorted(a_counts.keys(), reverse=True):
            count = a_counts[b]
            pct = count / a_total * 100
            print(f'  A{b}: {count:5d} channels ({pct:5.1f}%)')
    print('=' * 70)

    # Save configs
    weight_path = f'{base_name}_weight.json'
    act_path = f'{base_name}_activation.json'
    joint_path = f'{base_name}.json'

    # Joint config for reference
    joint_config = {
        'layers': {},
        'metadata': {
            'constraint': 'W=A (filter-level) with activation-disable fallback',
            'avg_weight_bits': round(w_avg, 3),
            'total_channels': w_total,
            'w_a_layers': n_act_layers,
            'weight_only_layers': n_wonly_layers,
            'baseline_accuracy': round(baseline_acc, 4),
            'target_drop': args.target_drop,
            'bit_choices': sorted(args.bits),
            'weight_bit_distribution': {f'W{b}': w_counts.get(b, 0)
                                         for b in sorted(args.bits, reverse=True)},
        }
    }
    for l, w_bits in weight_config.items():
        a_bits = activation_config.get(l, None)
        joint_config['layers'][l] = {
            'weight': w_bits,
            'activation': a_bits if a_bits is not None else 'disabled',
        }

    with open(weight_path, 'w') as f:
        json.dump(weight_config, f, indent=2)
    with open(act_path, 'w') as f:
        json.dump(activation_config, f, indent=2)
    with open(joint_path, 'w') as f:
        json.dump(joint_config, f, indent=2)

    print(f'\nSaved configs:')
    print(f'  Weight config:      {weight_path}')
    print(f'  Activation config:  {act_path}')
    print(f'  Joint config (ref): {joint_path}')
    if n_wonly_layers > 0:
        print(f'  Note: {n_wonly_layers} layers have activation quantization disabled')

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s')
    print(f'\nNext step: Run validate_config_granular.py with {weight_path}')


if __name__ == '__main__':
    main()
