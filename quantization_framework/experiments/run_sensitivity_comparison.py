"""
Sensitivity Metric Comparison Runner
=====================================
Runs all four sensitivity metrics through the full pipeline and produces
a comparison table. Each metric uses the same model, dataset, greedy search
hyperparameters, and PTQ validation — only the sensitivity scoring changes.

Metrics compared:
  1. ours      — RMS gradient norm (our method, channel_sensitivity.py)
  2. hawqv2    — Hessian trace per layer, broadcast to filters (NeurIPS 2020)
  3. hmqat     — Hessian trace × param count (Neural Networks 2025)
  4. l2grad    — Gradient L2 norm per filter (pre-Fix2 ablation baseline)

Pipeline per metric:
  Step 1: Generate sensitivity CSV
  Step 2: Run joint_granular_search.py  → weight + activation config JSON
  Step 3: Run validate_config_granular.py → PTQ accuracy, avg bits, BOPs

Usage — run on VGG-11/CIFAR-10:
    python run_sensitivity_comparison.py \\
        --model vgg11_bn \\
        --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 \\
        --bits 2 4 8 \\
        --target-drop 3.0 \\
        --output-dir comparison_results/vgg11

Usage — run on ResNet/GTSRB:
    python run_sensitivity_comparison.py \\
        --model resnet \\
        --checkpoint models/best_resnet_model_cifar100_changed.pth \\
        --dataset gtsrb \\
        --bits 2 4 8 \\
        --target-drop 3.0 \\
        --output-dir comparison_results/resnet

    Use --metrics to run a subset, e.g. --metrics ours hawqv2
    Use --skip-sensitivity to reuse existing CSVs (skip Step 1)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

# Locate scripts relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNEL_SENS   = os.path.join(SCRIPT_DIR, 'channel_sensitivity.py')
BASELINE_SENS  = os.path.join(SCRIPT_DIR, 'baseline_sensitivity_metrics.py')
JOINT_SEARCH   = os.path.join(SCRIPT_DIR, 'joint_granular_search.py')
VALIDATE       = os.path.join(SCRIPT_DIR, 'validate_config_granular.py')

ALL_METRICS = ['ours', 'hawqv2', 'gradnorm', 'fisher', 'mse', 'adabm', 'sensiboost', 'taylor', 'l2grad']
# hmqat excluded from default: formula unverified, effectively same as hawqv2


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run(cmd, label):
    print(f'\n{"=" * 65}')
    print(f'  {label}')
    print(f'{"=" * 65}')
    print(f'  CMD: {" ".join(cmd)}\n')
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f'\n[ERROR] Command failed (return code {result.returncode})')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parse results JSON from validate_config_granular.py
# ---------------------------------------------------------------------------

def parse_results(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    return {
        'ptq_acc':   data.get('ptq_accuracy',   data.get('accuracy', float('nan'))),
        'baseline':  data.get('baseline_accuracy', float('nan')),
        'drop':      data.get('accuracy_drop',  float('nan')),
        'avg_bits':  data.get('avg_bits',        float('nan')),
        'bops_red':  data.get('bops_reduction',  float('nan')),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run all sensitivity metrics and compare results'
    )
    parser.add_argument('--model', required=True,
                        choices=['vgg11_bn', 'resnet', 'levit', 'swin'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Max accuracy drop budget (δ, default 3.0%%)')
    parser.add_argument('--output-dir', default='comparison_results',
                        help='Directory for all output files')
    parser.add_argument('--metrics', nargs='+', default=ALL_METRICS,
                        choices=['ours','hawqv2','gradnorm','fisher','mse','adabm','sensiboost','taylor','l2grad','hmqat'],
                        help='Which metrics to run (default: all four)')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip Step 1 — reuse existing sensitivity CSVs')
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip Steps 1+2 — reuse existing sensitivity CSVs '
                             'AND config JSONs. Goes straight to PTQ validation.')
    parser.add_argument('--calib-batches', type=int, default=64)
    parser.add_argument('--n-vectors', type=int, default=50,
                        help='Hutchinson vectors per layer per batch for Hessian metrics. '
                             'HAWQ-V2 paper uses m=50. '
                             'VGG-11: 11 layers × 50 × 10 batches = 5500 grad calls. '
                             'Use 10 for faster runs, 50 to match paper exactly.')
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='Batches for greedy search PTQ eval (default: full val set)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)
    bits_str = [str(b) for b in args.bits]

    print('=' * 65)
    print('SENSITIVITY METRIC COMPARISON')
    print('=' * 65)
    print(f'  Model:       {args.model}')
    print(f'  Dataset:     {args.dataset}')
    print(f'  Bits:        {args.bits}')
    print(f'  Target drop: {args.target_drop}%')
    print(f'  Metrics:     {args.metrics}')
    print(f'  Output dir:  {args.output_dir}')
    print('=' * 65)

    results = {}
    t_total = time.time()

    for metric in args.metrics:
        print(f'\n\n{"#" * 65}')
        print(f'  METRIC: {metric.upper()}')
        print(f'{"#" * 65}')

        # File paths for this metric
        # joint_granular_search.py appends _weight.json and _activation.json
        # to whatever is passed as --output (base_name)
        sens_csv     = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{metric}_sensitivity.csv')
        act_err_json = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{metric}_act_errors.json')
        search_base  = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{metric}_config')
        weight_cfg   = search_base + '_weight.json'
        act_cfg_auto = search_base + '_activation.json'
        val_results  = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_{metric}_results.json')

        # ---- Step 1: Generate sensitivity CSV --------------------------------
        if args.skip_search:
            if not os.path.exists(weight_cfg):
                print(f'[SKIP-SEARCH] Config not found: {weight_cfg} — running full pipeline')
                args.skip_search = False  # fall through to normal flow
            else:
                print(f'[SKIP] Reusing existing configs for {metric}')

        if not args.skip_sensitivity and not args.skip_search:
            if metric == 'ours':
                run([
                    sys.executable, CHANNEL_SENS,
                    '--model', args.model,
                    '--checkpoint', args.checkpoint,
                    '--dataset', args.dataset,
                    '--bits', *bits_str,
                    '--output', sens_csv,
                    '--act-error-output', act_err_json,
                    '--calib-batches', str(args.calib_batches),
                ], f'Step 1/{metric}: RMS gradient norm sensitivity (our method)')
            else:
                baseline_cmd = [
                    sys.executable, BASELINE_SENS,
                    '--model', args.model,
                    '--checkpoint', args.checkpoint,
                    '--dataset', args.dataset,
                    '--metric', metric,
                    '--output', sens_csv,
                    '--calib-batches', str(args.calib_batches),
                    '--n-vectors', str(args.n_vectors),
                ]
                # l2grad also produces activation errors (same pipeline as ours)
                if metric == 'l2grad':
                    baseline_cmd += ['--act-error-output', act_err_json]
                run(baseline_cmd, f'Step 1/{metric}: {metric} sensitivity')
        else:
            if not os.path.exists(sens_csv):
                print(f'[SKIP] Sensitivity CSV not found: {sens_csv}')
                continue
            print(f'[SKIP] Reusing existing sensitivity CSV: {sens_csv}')

        # ---- Step 2: Greedy search -------------------------------------------
        if args.skip_search:
            print(f'[SKIP] Reusing greedy search configs: {weight_cfg}')
        else:
            search_cmd = [
                sys.executable, JOINT_SEARCH,
                '--model', args.model,
                '--checkpoint', args.checkpoint,
                '--sensitivity', sens_csv,
                '--dataset', args.dataset,
                '--bits', *bits_str,
                '--target-drop', str(args.target_drop),
                '--output', search_base,
            ]
            # Pass act-errors for 'ours' and 'l2grad' (both run full pipeline)
            if metric in ('ours', 'l2grad') and os.path.exists(act_err_json):
                search_cmd += ['--act-errors', act_err_json]
            if args.eval_batches:
                search_cmd += ['--eval-batches', str(args.eval_batches)]
            run(search_cmd, f'Step 2/{metric}: Greedy search')

        # ---- Step 3: PTQ validation -----------------------------------------
        # Use weight config for act config too if activation config missing
        act_cfg_for_val = act_cfg_auto if os.path.exists(act_cfg_auto) else weight_cfg
        run([
            sys.executable, VALIDATE,
            '--model', args.model,
            '--checkpoint', args.checkpoint,
            '--config', weight_cfg,
            '--activation-config', act_cfg_for_val,
            '--dataset', args.dataset,
            '--calib-batches', str(args.calib_batches),
            '--device', device,
            '--results-json', val_results,
        ], f'Step 3/{metric}: PTQ validation')

        # Parse results
        r = parse_results(val_results)
        if r:
            results[metric] = r
            print(f'\n  Result: PTQ={r["ptq_acc"]:.2f}%  Drop={r["drop"]:.2f}%  '
                  f'Bits={r["avg_bits"]:.2f}  BOPs={r["bops_red"]:.2f}x')

    # --------------------------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------------------------
    print(f'\n\n{"=" * 65}')
    print('  COMPARISON TABLE')
    print(f'{"=" * 65}')
    print(f'  Model: {args.model}   Dataset: {args.dataset}   δ={args.target_drop}%\n')

    metric_labels = {
        'hawqv2':   'HAWQ-V2 avg Hessian trace  (NeurIPS 2020)',
        'gradnorm': 'First-order grad norm       (ICCVW 2023)',
        'fisher':   'Diagonal Fisher per filter  (ICML 2023)',
        'mse':      'MSE gradient-free           (baseline)',
        'adabm':     'Activation std               (CVPR 2024)',
        'sensiboost':'SensiBoost norm. act. MSE    (arXiv 2503.06518, 2025)',
        'taylor':   'Taylor ||W⊙∂L/∂W||          (arXiv 2505.13060, 2025)',
        'l2grad':   'L2 gradient norm            (pre-Fix2 ablation)',
        'hmqat':    'HMQAT (unverified, ~HAWQ-V2)(NN 2025)',
        'ours':     'RMS gradient norm           (ours)',
    }

    header = f"{'Metric':<30} {'PTQ Acc':>9} {'Drop':>7} {'Avg Bits':>10} {'BOPs Red.':>11}"
    print(header)
    print('-' * len(header))

    for metric in ALL_METRICS:
        if metric not in results:
            print(f"  {metric_labels.get(metric, metric):<28}  [not run]")
            continue
        r = results[metric]
        marker = '  ◄ ours' if metric == 'ours' else ''
        print(
            f"  {metric_labels.get(metric, metric):<28}"
            f"  {r['ptq_acc']:>7.2f}%"
            f"  {r['drop']:>5.2f}%"
            f"  {r['avg_bits']:>8.2f}"
            f"  {r['bops_red']:>9.2f}x"
            f"{marker}"
        )

    print(f'\n  Baseline (FP32): {results[args.metrics[0]]["baseline"]:.2f}%'
          if args.metrics and args.metrics[0] in results else '')
    print(f'\n  Full results saved in: {args.output_dir}/')

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'model': args.model,
            'dataset': args.dataset,
            'target_drop': args.target_drop,
            'bits': args.bits,
            'results': results,
        }, f, indent=2)
    print(f'  Summary JSON: {summary_path}')
    print(f'\n  Total time: {(time.time() - t_total) / 60:.1f} min')


if __name__ == '__main__':
    main()
