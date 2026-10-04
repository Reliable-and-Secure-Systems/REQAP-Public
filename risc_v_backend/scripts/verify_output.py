"""
verify_output.py
================
Task 3 — Python verification script (all models)

Computes the same MAC dot products that main_inference.c computes,
using the raw weight values from the .pth file and the MQF JSON config.

This is the "ground truth" side of the test.
If Python results match C results → the SWAR kernel is correct.

Usage:
  cd Prune_2
  python risc_v_backend/verify_output.py --model resnet18
  python risc_v_backend/verify_output.py --model alexnet
  python risc_v_backend/verify_output.py --model vgg11

What it does:
  1. Loads the model from the .pth file (same injection logic as generate_packed_c.py)
  2. For each test layer, extracts weights, quantizes to MQF bit-width
  3. Uses activation = 1 everywhere (same as main_inference.c)
  4. Computes dot product = sum of quantized weights (since act=1)
  5. Prints results to compare side-by-side with C output
"""

import os
import sys
import types
import argparse
import numpy as np

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SIM_DIR      = os.path.join(PROJECT_ROOT, "systolic_sim")

for p in [PROJECT_ROOT, SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn


# ── Model registry (mirrors generate_packed_c.py) ────────────────────────────

MODEL_REGISTRY = {
    "alexnet": {
        "pth":   os.path.join(PROJECT_ROOT, "models", "qalex-8bit.pth"),
        "class": "AlexNet",
        # One representative layer per vector type present in the config
        "test_layers": [
            # (layer_name,  bits,  type_label,  d)
            ("conv1.0",     8,     "8bit-d1",   1),   # K=121,  n_words=121
            ("conv2.0",     8,     "8bit-d1",   1),   # K=2400, baseline
            ("conv3.0",     8,     "8bit-d1",   1),   # K=2304, baseline, uniform 8b
        ],
    },
    "vgg11": {
        "pth":   os.path.join(PROJECT_ROOT, "models", "qvgg-8bit.pth"),
        "class": "vgg11_bn",
        "test_layers": [
            ("features.0",  8,   "8bit-d1",   1),   # K=27
            ("features.4",  8,   "8bit-d1",   1),   # K=576
            ("features.8",  8,   "8bit-d1",   1),   # K=1152
            ("classifier.0",8,   "8bit-d1",   1),   # K=512
            ("features.25", 2,   "general-d8",8),   # K=4608
        ],
    },
    "resnet18": {
        "pth":   os.path.join(PROJECT_ROOT, "models", "qresnet-8bit.pth"),
        "class": "ResNet18",
        "test_layers": [
            ("conv1",            8,  "8bit-d1",    1),  # K=27
            ("layer1.0.conv1",   8,  "8bit-d1",    1),  # K=64
            ("layer1.0.conv3",   8,  "8bit-d1",    1),  # K=64
            ("layer1.0.conv2",   4,  "4bit-d4",    4),  # K=576, n_words=144
            ("layer1.1.conv2",   4,  "4bit-d4",    4),  # K=576, n_words=144
            # Test C (general): same layer as 4bit-d4 row 1, different C path
            ("layer1.0.conv2",   4,  "general-d4", 4),
        ],
    },
}


# ── Quantization helper (identical to generate_packed_c.py) ──────────────────

def quantize_weight(value: float, bits: int) -> int:
    """Symmetric quantization: float → signed int in [-(2^(b-1)), 2^(b-1)-1]."""
    if bits <= 0:
        return 0
    max_int = (1 << (bits - 1)) - 1
    min_int = -(1 << (bits - 1))
    scaled  = value * max_int
    return int(max(min_int, min(max_int, round(scaled))))


# ── Model loader (identical injection logic to generate_packed_c.py) ─────────

def load_model(model_name: str) -> nn.Module:
    """
    Load model with the same class injection used in generate_packed_c.py
    so that torch.load can unpickle .pth files saved with the full model object.
    """
    from quantization_framework.models.alexnet import AlexNet
    from quantization_framework.models.vgg import VGG, vgg11_bn
    from quantization_framework.models.resnet import ResNet, ResNet18, BasicBlock, Bottleneck
    import __main__

    # Inject under __main__ (needed for .pth files pickled from __main__)
    __main__.fasion_mnist_alexnet = AlexNet
    __main__.AlexNet              = AlexNet
    __main__.VGG                  = VGG
    __main__.vgg11_bn             = vgg11_bn
    __main__.ResNet               = ResNet
    __main__.ResNet18             = ResNet18
    __main__.BasicBlock           = BasicBlock
    __main__.Bottleneck           = Bottleneck

    # Inject into sys.modules (needed for .pth files pickled from a named module)
    fake_resnet = types.ModuleType("models.resnet")
    fake_resnet.ResNet     = ResNet
    fake_resnet.ResNet18   = ResNet18
    fake_resnet.BasicBlock = BasicBlock
    fake_resnet.Bottleneck = Bottleneck
    sys.modules.setdefault("models", types.ModuleType("models"))
    sys.modules["models.resnet"] = fake_resnet

    fake_vgg = types.ModuleType("models.vgg")
    fake_vgg.VGG      = VGG
    fake_vgg.vgg11_bn = vgg11_bn
    sys.modules["models.vgg"] = fake_vgg

    pth_path = MODEL_REGISTRY[model_name]["pth"]
    print(f"  Loading: {os.path.basename(pth_path)}")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.checkpoint_dict = model.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = (AlexNet(num_classes=10) if model_name == "alexnet"
                 else vgg11_bn(num_classes=10) if model_name == "vgg11"
                 else ResNet18(num_classes=43))
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.checkpoint_dict = checkpoint["state_dict"]
    else:
        model = (AlexNet(num_classes=10) if model_name == "alexnet"
                 else vgg11_bn(num_classes=10) if model_name == "vgg11"
                 else ResNet18(num_classes=43))
        try:
            model.load_state_dict(checkpoint, strict=False)
        except Exception:
            pass  # Use loaded weights as-is if state_dict keys differ
        model.checkpoint_dict = checkpoint

    # Ensure weight.data holds the already-quantized values if present
    state_dict = model.checkpoint_dict
    for name, module in model.named_modules():
        w_data_key = f"{name}.weight._data"
        if w_data_key in state_dict:
            module.weight.data = state_dict[w_data_key].float()

    model.eval()
    return model


# ── Weight extraction (identical to generate_packed_c.py) ────────────────────

def get_weight_vec(model: nn.Module, layer_name: str) -> np.ndarray:
    """
    Extract and flatten weights for a named layer.
    """
    state_dict = getattr(model, "checkpoint_dict", {})
    w_data_key = f"{layer_name}.weight._data"
    if w_data_key in state_dict:
        w_raw = state_dict[w_data_key].float().view(-1).numpy()
        return w_raw / 127.0
        
    for name, module in model.named_modules():
        if name != layer_name:
            continue
        w = module.weight.data
        w_flat = w.view(-1).numpy()
        w_max = float(np.abs(w_flat).max()) or 1.0
        return w_flat / (w_max + 1e-8)
    raise ValueError(f"Layer '{layer_name}' not found in model")


# ── MAC compute (with activation = 1) ────────────────────────────────────────

def compute_mac(w_vec: np.ndarray, bits: int) -> int:
    """
    Quantize each weight to `bits` bits, then sum them.
    With activation = 1 everywhere: dot_product = sum(quantized_weights).
    This is the Python ground truth for both swar_mac_8bit and swar_mac_4bit_d4.
    """
    total = 0
    for v in w_vec:
        q = quantize_weight(float(v), bits)
        if bits < 8 and bits > 0:
            max_int = (1 << (bits - 1)) - 1
            if max_int > 0:
                q = int(q * 127 / max_int)
        total += q
    return total


def n_words_for(layer_name: str, w_vec: np.ndarray, d: int, bits: int) -> int:
    """Return n_words as the C test harness sees it."""
    if d == 1:
        return len(w_vec)          # one weight per word
    return len(w_vec) // d        # d weights packed per word


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Python ground truth for SWAR MAC kernel (Task 3)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="resnet18",
                        help="Model to verify (default: resnet18)")
    args = parser.parse_args()

    model_name = args.model
    info       = MODEL_REGISTRY[model_name]

    print(f"\n[1/2] Loading {model_name} model...")
    model = load_model(model_name)

    print(f"\n[2/2] Computing MAC ground truth...")
    print()
    print("=" * 65)
    print(f"  Python Ground Truth — {model_name.upper()}  (activation = 1)")
    print(f"  Compare with: ./test_inference --model {model_name}")
    print("=" * 65)
    print()

    for layer_name, bits, type_label, d in info["test_layers"]:
        w_vec   = get_weight_vec(model, layer_name)
        result  = compute_mac(w_vec, bits)
        nw      = n_words_for(layer_name, w_vec, d, bits)
        print(f"  [{layer_name:<25}] type={type_label:<11} "
              f"n_words={nw:4d}  d={d}  result={result}")

    print()
    print("=" * 65)
    print("  All results above must match the C test_inference output.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
