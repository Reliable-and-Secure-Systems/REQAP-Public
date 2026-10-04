"""
Auto-Quantization Engine: Filter-Level W=A Co-Optimization
===========================================================

Orchestrates the full filter-level (per-output-channel) quantization pipeline:

  Step 1: Per-filter sensitivity analysis (weight quantization error + activation norms)
  Step 2: Greedy W=A filter-level search
  Step 2.5: W=A constraint verification
  Step 3: PTQ gate — validate accuracy on full validation set
  Step 4: BOPs analysis
  Step 5: Save metrics.json

W=A constraint: each output channel's weight and the corresponding output
activation channel always share the same bit-width.

Comparison with layer-wise engine:
  - auto_quantize_engine_joint.py:   1 bit-width per layer (~minutes profiling)
  - auto_quantize_engine_granular.py: 1 bit-width per filter (~seconds profiling)

Usage:
    python auto_quantize_engine_granular.py \\
        --model vgg11_bn \\
        --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 \\
        --bits 2 4 8 \\
        --target-drop 3.0
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader,
)
from models.model_loaders import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd):
    """Run a shell command and raise on failure."""
    print(f'\n[ENGINE] Running: {cmd}')
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f'Command failed (exit {ret}): {cmd}')


def get_dataloader(dataset, model_name, batch_size=None):
    if batch_size is None:
        if model_name == 'swin':
            batch_size = 16
        elif model_name == 'levit':
            batch_size = 32
        else:
            batch_size = 128
    input_size = 224 if model_name in ('levit', 'swin') or dataset == 'gtsrb' else 32
    if dataset == 'cifar10':
        return get_cifar10_dataloader(train=False, batch_size=batch_size, input_size=input_size)
    elif dataset == 'cifar100':
        return get_cifar100_dataloader(train=False, batch_size=batch_size, input_size=input_size)
    elif dataset == 'gtsrb':
        return get_gtsrb_dataloader(train=False, batch_size=batch_size, input_size=input_size)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def verify_wa_constraint(weight_config, activation_config):
    """Return (is_compliant, n_wa_matching, n_wa_layers, n_weight_only).

    Layers present in both configs must have identical bit-lists (W=A).
    Layers absent from activation_config are weight-only (intentional, not a violation).
    """
    matching = 0
    wa_total = 0
    weight_only = 0
    for layer in weight_config:
        if layer in activation_config:
            wa_total += 1
            if weight_config[layer] == activation_config[layer]:
                matching += 1
        else:
            weight_only += 1
    return matching == wa_total, matching, wa_total, weight_only


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def auto_quantize_granular(model_name, checkpoint_path, dataset='cifar10',
                            bit_choices=None, target_drop=3.0,
                            qat_threshold=5.0, qat_epochs=5, qat_lr=1e-4,
                            output_metrics='metrics_granular.json',
                            weight_only=False, save_model=None):
    if bit_choices is None:
        bit_choices = [2, 4, 8]

    bits_str = '_'.join(map(str, sorted(bit_choices)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100 if dataset == 'cifar100' else (43 if dataset == 'gtsrb' else 10)

    # File paths
    sensitivity_csv = f'{model_name}_{dataset}_channel_sensitivity.csv'
    act_errors_json = f'{model_name}_{dataset}_activation_errors.json'
    weight_cfg = f'{model_name}_{dataset}_channel_config_{bits_str}_weight.json'
    act_cfg = f'{model_name}_{dataset}_channel_config_{bits_str}_activation.json'
    joint_cfg = f'{model_name}_{dataset}_channel_config_{bits_str}.json'
    ptq_results_json = f'{model_name}_{dataset}_ptq_results.json'

    # Script paths (relative to this file's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sensitivity_script = os.path.join(script_dir, 'channel_sensitivity.py')
    search_script = os.path.join(script_dir, 'joint_granular_search.py')
    validate_script = os.path.join(script_dir, 'validate_config_granular.py')
    qat_script = os.path.join(script_dir, 'qat_training.py')

    bits_arg = ' '.join(map(str, bit_choices))

    print('=' * 70)
    print('AUTO-QUANTIZATION ENGINE (FILTER-LEVEL W=A CO-OPTIMIZATION)')
    print('=' * 70)
    print(f'Model:        {model_name}')
    print(f'Dataset:      {dataset}')
    print(f'Bit-Choices:  {bit_choices}')
    print(f'Constraint:   W=A enforced per output channel (filter-level)')
    print(f'Target Drop:  {target_drop}%')
    print('=' * 70)

    timings = {}
    t_total = time.time()

    # ------------------------------------------------------------------
    # STEP 1: Per-filter sensitivity analysis
    # ------------------------------------------------------------------
    if os.path.exists(sensitivity_csv):
        print(f'\n[STEP 1] Sensitivity CSV already exists — skipping ({sensitivity_csv})')
        timings['sensitivity'] = 0.0
    else:
        print(f'\n[STEP 1] Computing per-filter sensitivity ({sensitivity_csv})...')
        t1 = time.time()
        sens_cmd = (
            f'python {sensitivity_script} '
            f'--model {model_name} '
            f'--checkpoint {checkpoint_path} '
            f'--dataset {dataset} '
            f'--bits {bits_arg} '
            f'--output {sensitivity_csv}'
        )
        if not weight_only:
            # Activation errors are only needed for the activation guard / rollback
            sens_cmd += f' --act-error-output {act_errors_json}'
        run_command(sens_cmd)
        timings['sensitivity'] = time.time() - t1
        print(f'[TIMING] Sensitivity completed in {timings["sensitivity"]:.1f}s '
              f'({timings["sensitivity"]/60:.1f} min)')

    # ------------------------------------------------------------------
    # STEP 2: Greedy filter-level W=A search
    # ------------------------------------------------------------------
    print(f'\n[STEP 2] Greedy W=A filter-level search...')
    t2 = time.time()
    search_cmd = (
        f'python {search_script} '
        f'--model {model_name} '
        f'--checkpoint {checkpoint_path} '
        f'--sensitivity {sensitivity_csv} '
        f'--dataset {dataset} '
        f'--bits {bits_arg} '
        f'--target-drop {target_drop} '
        f'--output {model_name}_{dataset}_channel_config_{bits_str}'
    )
    if weight_only:
        search_cmd += ' --weight-only'
    else:
        if os.path.exists(act_errors_json):
            search_cmd += f' --act-errors {act_errors_json}'
        # Enable accuracy-aware rollback: validate during search, bump layers if needed.
        search_cmd += f' --max-drop {target_drop}'
    run_command(search_cmd)
    timings['search'] = time.time() - t2
    print(f'[TIMING] Search completed in {timings["search"]:.1f}s')

    # ------------------------------------------------------------------
    # STEP 2.5: Verify W=A constraint
    # ------------------------------------------------------------------
    print(f'\n[STEP 2.5] Verifying W=A constraint...')
    with open(weight_cfg, 'r') as f:
        w_config = json.load(f)
    with open(act_cfg, 'r') as f:
        a_config = json.load(f)

    compliant, n_match, n_wa, n_wonly = verify_wa_constraint(w_config, a_config)
    total_ch = sum(len(v) if isinstance(v, list) else 1 for v in w_config.values())
    all_bits = [b for v in w_config.values() for b in (v if isinstance(v, list) else [v])]
    avg_bits = sum(all_bits) / len(all_bits) if all_bits else 0

    print(f'  W=A layers:     {n_match}/{n_wa} compliant')
    if n_wonly > 0:
        reason = '--weight-only flag' if weight_only else 'accuracy-aware rollback'
        print(f'  Weight-only:    {n_wonly} layers (activation quant disabled by {reason})')
    print(f'  Total channels: {total_ch}  |  Avg weight bits: {avg_bits:.2f}')

    if not compliant:
        print('[WARNING] W=A constraint not fully satisfied for W+A layers — proceeding anyway')
    else:
        print('[PASSED] W=A constraint satisfied (+ weight-only layers where needed)')

    # ------------------------------------------------------------------
    # STEP 3: PTQ gate
    # ------------------------------------------------------------------
    print(f'\n[STEP 3] PTQ Validation (filter-level)...')
    t3 = time.time()

    # Measure baseline accuracy first
    print('  Measuring FP32 baseline...')
    model_fp32 = load_model(model_name, checkpoint_path=checkpoint_path,
                             num_classes=num_classes)
    model_fp32 = model_fp32.to(device)
    loader = get_dataloader(dataset, model_name)
    baseline_acc = evaluate_accuracy(model_fp32, loader, device=device)
    del model_fp32
    print(f'  Baseline: {baseline_acc:.2f}%')

    # Sanity gate: a correctly-loaded, correctly-paired model/dataset should
    # score well above random chance. This catches wrong-dataset /
    # wrong-checkpoint pairings that don't raise an exception (e.g. a
    # checkpoint that loads cleanly but was trained for a different task) —
    # complements the hard-fail in model_loaders.py. See
    # REMEDIATION_PLAN.md Phase A4 / audit Finding 5.
    random_chance = 100.0 / num_classes
    if baseline_acc < 3 * random_chance:
        raise RuntimeError(
            f'FP32 baseline accuracy ({baseline_acc:.2f}%) is barely above random '
            f'chance ({random_chance:.2f}% for {num_classes} classes) — this almost '
            f'certainly means the wrong checkpoint/dataset/num_classes were paired '
            f'(model={model_name!r}, dataset={dataset!r}, checkpoint={checkpoint_path!r}). '
            f'Aborting before wasting a search+validation run on a broken baseline.'
        )

    validate_cmd = (
        f'python {validate_script} '
        f'--model {model_name} '
        f'--checkpoint {checkpoint_path} '
        f'--config {weight_cfg} '
        f'--activation-config {act_cfg} '
        f'--dataset {dataset} '
        f'--results-json {ptq_results_json}'
    )
    if save_model:
        validate_cmd += f' --save-model {save_model}'
    run_command(validate_cmd)
    timings['validation'] = time.time() - t3
    print(f'[TIMING] Validation completed in {timings["validation"]:.1f}s')

    # ------------------------------------------------------------------
    # STEP 3.5: QAT gate — recover if PTQ accuracy drop is too large
    # ------------------------------------------------------------------
    ptq_acc = None
    if os.path.exists(ptq_results_json):
        with open(ptq_results_json, 'r') as f:
            ptq_results = json.load(f)
        ptq_acc = ptq_results.get('ptq_accuracy')

    if ptq_acc is not None:
        ptq_drop = baseline_acc - ptq_acc
        print(f'\n[STEP 3.5] PTQ drop: {ptq_drop:.2f}% '
              f'(baseline {baseline_acc:.2f}% → PTQ {ptq_acc:.2f}%)')
        if ptq_drop > qat_threshold:
            print(f'[QAT GATE] Drop {ptq_drop:.2f}% > threshold {qat_threshold:.1f}% '
                  f'— triggering QAT recovery...')
            qat_out = f'{model_name}_qat_recovered.pth'
            t_qat = time.time()
            run_command(
                f'python {qat_script} '
                f'--model {model_name} '
                f'--checkpoint {checkpoint_path} '
                f'--config {weight_cfg} '
                f'--activation-config {act_cfg} '
                f'--dataset {dataset} '
                f'--epochs {qat_epochs} '
                f'--lr {qat_lr} '
                f'--output {qat_out}'
            )
            timings['qat'] = time.time() - t_qat
            print(f'[TIMING] QAT completed in {timings["qat"]:.1f}s '
                  f'({timings["qat"]/60:.1f} min)')
            print(f'[QAT GATE] Fine-tuned checkpoint saved to {qat_out}')
        else:
            print(f'[QAT GATE] Drop within threshold — PTQ result accepted, no QAT needed.')
    else:
        print('\n[STEP 3.5] PTQ results JSON not found — skipping QAT gate check.')

    # ------------------------------------------------------------------
    # STEP 4: BOPs
    # ------------------------------------------------------------------
    # Read BOPs from joint config metadata if available
    if os.path.exists(joint_cfg):
        with open(joint_cfg, 'r') as f:
            joint_data = json.load(f)
        meta = joint_data.get('metadata', {})
        print(f'\n[STEP 4] BOPs from config metadata:')
        reported_avg = meta.get('avg_weight_bits', meta.get('avg_bits', avg_bits))
        print(f'  Avg bits: {reported_avg:.2f}')

    # ------------------------------------------------------------------
    # STEP 5: Save metrics
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t_total

    metrics = {
        'model': model_name,
        'dataset': dataset,
        'granularity': 'filter-level (per-output-channel)',
        'constraint': 'weight-only (mixed-precision)' if weight_only else 'W=A (filter-level)',
        'bit_choices': sorted(bit_choices),
        'target_drop': target_drop,
        'qat_threshold': qat_threshold,
        'baseline_accuracy': round(baseline_acc, 4),
        'ptq_accuracy': round(ptq_acc, 4) if ptq_acc is not None else None,
        'average_bits': round(avg_bits, 3),
        'total_channels': total_ch,
        'wa_compliant_layers': n_match,
        'wa_total_layers': n_wa,
        'weight_only_layers': n_wonly,
        'timings': {k: round(v, 2) for k, v in timings.items()},
        'total_time_s': round(total_elapsed, 1),
        'configs': {
            'weight': weight_cfg,
            'activation': act_cfg,
            'joint': joint_cfg,
        }
    }
    with open(output_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n' + '=' * 70)
    print('TIMING SUMMARY')
    print('=' * 70)
    for step, t in timings.items():
        print(f'  {step:<20}: {t:>8.1f}s ({t/60:.1f} min)')
    print(f'  {"-" * 40}')
    print(f'  Total               : {total_elapsed:>8.1f}s ({total_elapsed/60:.1f} min)')
    print('=' * 70)
    print(f'\n[METRICS] Saved to {output_metrics}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Filter-level W=A co-optimized quantization engine'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model: levit, resnet, swin, vgg11_bn')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Max allowed accuracy drop %% (same scale as layer-wise engine)')
    parser.add_argument('--qat-threshold', type=float, default=5.0,
                        help='Trigger QAT if PTQ accuracy drop exceeds this %%')
    parser.add_argument('--qat-epochs', type=int, default=5,
                        help='QAT fine-tuning epochs (used only when QAT gate triggers)')
    parser.add_argument('--qat-lr', type=float, default=1e-4,
                        help='QAT learning rate')
    parser.add_argument('--output-metrics', type=str, default=None)
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save final quantized model checkpoint (.pth)')
    parser.add_argument('--weight-only', action='store_true',
                        help='Disable activation quantization. Greedy search assigns '
                             'weight bit-widths normally; activation config is empty. '
                             'Use for architectures where activation quantization is '
                             'not feasible (e.g. Swin Transformer).')
    args = parser.parse_args()

    metrics_file = args.output_metrics or f'{args.model}_{args.dataset}_metrics_granular.json'

    auto_quantize_granular(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        bit_choices=args.bits,
        target_drop=args.target_drop,
        qat_threshold=args.qat_threshold,
        qat_epochs=args.qat_epochs,
        qat_lr=args.qat_lr,
        output_metrics=metrics_file,
        weight_only=args.weight_only,
        save_model=args.save_model,
    )


if __name__ == '__main__':
    main()
