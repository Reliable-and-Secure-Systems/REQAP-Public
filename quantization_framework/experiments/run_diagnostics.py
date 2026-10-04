"""
Diagnostic verification runs for Table 6.

Runs two experiments for each model:
  1. Weight-only  : all layers quantized (8-bit weights), NO activation quantization
  2. W8/A8 uniform: all layers at 8-bit weights AND 8-bit activations

Run from the Paper2 root directory on the server:
    python run_diagnostics.py

Results are saved to:
    diag_swin_weight_only_results.json
    diag_swin_w8a8_results.json
    diag_levit100_weight_only_results.json
    diag_levit100_w8a8_results.json
"""

import json
import os
import subprocess
import sys

import torch
import torch.nn as nn

# validate_config_granular.py lives in quantization_framework/experiments/
# and adds quantization_framework/ to sys.path — we replicate that here.
_QF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization_framework")
sys.path.insert(0, _QF)
from models.model_loaders import load_model


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def build_all8_config(model):
    """
    Build a weight config with every channel set to 8-bit.
    Returns {layer_name: [8, 8, ..., 8]} for all Conv2d and Linear layers.
    """
    config = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            n_channels = module.weight.shape[0]
            config[name] = [8] * n_channels
    return config


def build_empty_config():
    """Empty activation config — no activation quantizers inserted (weight-only)."""
    return {}


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_experiment(label, model_name, checkpoint, dataset,
                   weight_cfg_path, act_cfg_path, results_path):
    print()
    print("=" * 70)
    print(f"RUNNING: {label}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "quantization_framework/experiments/validate_config_granular.py",
        "--model",            model_name,
        "--checkpoint",       checkpoint,
        "--config",           weight_cfg_path,
        "--activation-config", act_cfg_path,
        "--dataset",          dataset,
        "--results-json",     results_path,
    ]
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] {label} exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    experiments = [
        {
            "label":      "Swin-T CIFAR-100  — Weight-only",
            "model":      "swin",
            "checkpoint": "models/best_swin_model_cifar_changed.pth",
            "dataset":    "cifar100",
            "num_classes": 100,
            "weight_cfg": "diag_swin_w8_weight.json",
            "act_cfg":    "diag_swin_empty_act.json",
            "results":    "diag_swin_weight_only_results.json",
        },
        {
            "label":      "Swin-T CIFAR-100  — W8/A8 uniform",
            "model":      "swin",
            "checkpoint": "models/best_swin_model_cifar_changed.pth",
            "dataset":    "cifar100",
            "num_classes": 100,
            "weight_cfg": "diag_swin_w8_weight.json",   # same weight config
            "act_cfg":    "diag_swin_w8_act.json",
            "results":    "diag_swin_w8a8_results.json",
        },
        {
            "label":      "LeViT-384 CIFAR-100 — Weight-only",
            "model":      "levit",
            "checkpoint": "models/best_levit_model_cifar100.pth",
            "dataset":    "cifar100",
            "num_classes": 100,
            "weight_cfg": "diag_levit100_w8_weight.json",
            "act_cfg":    "diag_levit100_empty_act.json",
            "results":    "diag_levit100_weight_only_results.json",
        },
        {
            "label":      "LeViT-384 CIFAR-100 — W8/A8 uniform",
            "model":      "levit",
            "checkpoint": "models/best_levit_model_cifar100.pth",
            "dataset":    "cifar100",
            "num_classes": 100,
            "weight_cfg": "diag_levit100_w8_weight.json",   # same weight config
            "act_cfg":    "diag_levit100_w8_act.json",
            "results":    "diag_levit100_w8a8_results.json",
        },
    ]

    # -----------------------------------------------------------------------
    # Step 1: generate config JSONs (load each model once)
    # -----------------------------------------------------------------------
    print("\n[STEP 1] Generating all-8-bit config files...")

    # --- Swin ---
    print("  Loading Swin-T (CIFAR-100)...")
    swin_model = load_model("swin", "models/best_swin_model_cifar_changed.pth",
                            num_classes=100)
    swin_w8 = build_all8_config(swin_model)

    with open("diag_swin_w8_weight.json", "w") as f:
        json.dump(swin_w8, f)
    with open("diag_swin_w8_act.json", "w") as f:
        json.dump(swin_w8, f)          # same as weight for W8/A8
    with open("diag_swin_empty_act.json", "w") as f:
        json.dump({}, f)               # empty → no activation quantizers

    total = sum(len(v) for v in swin_w8.values())
    print(f"  Swin configs saved — {len(swin_w8)} layers, {total} channels")
    del swin_model

    # --- LeViT ---
    print("  Loading LeViT-384 (CIFAR-100)...")
    levit_model = load_model("levit", "models/best_levit_model_cifar100.pth",
                             num_classes=100)
    levit_w8 = build_all8_config(levit_model)

    with open("diag_levit100_w8_weight.json", "w") as f:
        json.dump(levit_w8, f)
    with open("diag_levit100_w8_act.json", "w") as f:
        json.dump(levit_w8, f)
    with open("diag_levit100_empty_act.json", "w") as f:
        json.dump({}, f)

    total = sum(len(v) for v in levit_w8.values())
    print(f"  LeViT configs saved — {len(levit_w8)} layers, {total} channels")
    del levit_model

    # -----------------------------------------------------------------------
    # Step 2: run all four experiments
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Running experiments...")
    for exp in experiments:
        run_experiment(
            label        = exp["label"],
            model_name   = exp["model"],
            checkpoint   = exp["checkpoint"],
            dataset      = exp["dataset"],
            weight_cfg_path = exp["weight_cfg"],
            act_cfg_path    = exp["act_cfg"],
            results_path    = exp["results"],
        )

    # -----------------------------------------------------------------------
    # Step 3: print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    for exp in experiments:
        path = exp["results"]
        if os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
            acc  = res.get("ptq_accuracy", res.get("accuracy", "N/A"))
            drop = res.get("accuracy_drop", "N/A")
            print(f"  {exp['label']:<45}  acc={acc}%  drop={drop}%")
        else:
            print(f"  {exp['label']:<45}  [no results file]")

    print()
    print("Results saved to diag_swin_*.json and diag_levit100_*.json")


if __name__ == "__main__":
    main()
