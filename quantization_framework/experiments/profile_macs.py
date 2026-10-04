"""
Ground-truth per-layer MAC counts for the CIFAR/GTSRB models, using
torch's built-in FlopCounterMode (no new dependency — replaces the
hand-rolled spatial/token tracking in compute_bops_filter_level, which
undercounts MaxPool-downsampled convs and ignores the Linear token
dimension entirely; see REMEDIATION_PLAN.md Phase A1).

Usage:
    python profile_macs.py --model vgg11_bn --dataset cifar10
    python profile_macs.py --all
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model

# (model_name, dataset, num_classes, checkpoint) — mirrors the 5 CIFAR/GTSRB
# configs in the paper (see MEMORY.md checkpoints table).
CONFIGS = [
    ('vgg11_bn', 'cifar10',  10,  'models/vgg11_bn.pt'),
    ('resnet',   'gtsrb',    43,  'models/resnet_gtsrb_new.pth'),
    ('levit',    'cifar10',  10,  'models/best3_levit_model_cifar10.pth'),
    ('levit',    'cifar100', 100, 'models/best_levit_model_cifar100.pth'),
    ('swin',     'cifar100', 100, 'models/best_swin_model_cifar_changed.pth'),
]

# Published reference GMACs to sanity-check against (approximate, from
# public model-zoo / paper reports for the closest matching architecture
# and input resolution — VGG-11/CIFAR and GTSRB@224 don't have a single
# canonical published number, so those two are order-of-magnitude checks).
REFERENCE_GMACS = {
    'vgg11_bn_cifar10':  0.15,   # VGG-11 @ 32x32, order-of-magnitude
    'resnet_gtsrb':      1.8,    # ResNet-18 @ 224x224, order-of-magnitude
    'levit_cifar10':     2.35,   # LeViT-384 @ 224x224 (published)
    'levit_cifar100':    2.35,
    'swin_cifar100':     4.5,    # Swin-T @ 224x224 (published)
}


def input_size_for(model_name, dataset):
    return 224 if model_name in ('levit', 'swin') or dataset == 'gtsrb' else 32


def profile_one(model_name, dataset, num_classes, checkpoint, root):
    ckpt_path = os.path.join(root, checkpoint)
    model = load_model(model_name, checkpoint_path=ckpt_path, num_classes=num_classes)
    model.eval()

    size = input_size_for(model_name, dataset)
    x = torch.randn(1, 3, size, size)

    # Leaf Conv2d/Linear names (the ones that actually get quantized —
    # matches the naming convention in the *_weight.json config files).
    leaf_names = {
        name for name, mod in model.named_modules()
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))
    }

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)

    # get_flop_counts() keys are "<RootClassName>.<dotted.path>" and include
    # BOTH leaf modules and their container aggregates (e.g. both
    # "VGG.features.0" and "VGG.features" and "VGG" itself, where the
    # containers double-count their children's MACs). Strip the root
    # prefix and keep only entries matching a real Conv2d/Linear leaf —
    # this naturally drops every container aggregate.
    per_module = fc.get_flop_counts()
    total_macs = fc.get_total_flops()

    layer_macs = {}
    for qualname, ops in per_module.items():
        name = qualname.split('.', 1)[1] if '.' in qualname else ''
        if name in leaf_names:
            layer_macs[name] = sum(v for v in ops.values())

    # Compute not attributed to any quantized Conv2d/Linear leaf (e.g.
    # attention QK^T / softmax@V matmuls done as raw ops inside a module's
    # forward()) — this always executes at FP32 and must still be counted
    # in both the FP32 and quantized BOPs totals (see REMEDIATION_PLAN.md
    # Phase A1 / audit Finding 2).
    other_macs = total_macs - sum(layer_macs.values())

    key = f'{model_name}_{dataset}'
    ref = REFERENCE_GMACS.get(key)
    total_gmacs = total_macs / 1e9
    print(f'{key}: {total_gmacs:.3f} GMACs'
          + (f'  (reference ~{ref} GMACs)' if ref else ''))
    if ref and not (0.3 * ref <= total_gmacs <= 3 * ref):
        print(f'  [WARN] total is >3x off from the order-of-magnitude reference — check model/input size')

    return {
        'layer_macs': layer_macs,
        'other_macs': other_macs,
        'total_macs': total_macs,
        'input_size': size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vgg11_bn', 'resnet', 'levit', 'swin'])
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--all', action='store_true')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--output-dir', default=os.path.join(root, 'mac_profiles'),
                        help='Shared across both pipelines — default is <repo_root>/mac_profiles')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        configs = CONFIGS
    else:
        if not args.model or not args.dataset:
            parser.error('specify --model and --dataset, or --all')
        match = [c for c in CONFIGS if c[0] == args.model and c[1] == args.dataset]
        if not match:
            parser.error(f'no config for {args.model}/{args.dataset}')
        configs = match

    for model_name, dataset, num_classes, checkpoint in configs:
        result = profile_one(model_name, dataset, num_classes, checkpoint, root)
        out_path = os.path.join(args.output_dir, f'{model_name}_{dataset}_macs.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
