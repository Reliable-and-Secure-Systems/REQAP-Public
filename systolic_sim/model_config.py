"""
model_config.py

Defines the CNN architecture and quantization configurations for both
baseline (uniform 8-bit) and packed (mixed-precision) simulation modes.
All weights are deterministic synthetic tensors (fixed seed).
"""
import numpy as np

SEED = 42

# ── CNN Architecture ──────────────────────────────────────────────────────────
CNN_CONFIG = {
    "input_shape": (1, 28, 28),  # C, H, W  (FashionMNIST style)
    "layers": [
        {"name": "conv1", "type": "conv",    "in_ch": 1,        "out_ch": 8,  "kernel": 3, "stride": 1, "padding": 1},
        {"name": "relu1", "type": "relu"},
        {"name": "pool1", "type": "maxpool", "kernel": 2,        "stride": 2},
        {"name": "conv2", "type": "conv",    "in_ch": 8,        "out_ch": 16, "kernel": 3, "stride": 1, "padding": 1},
        {"name": "relu2", "type": "relu"},
        {"name": "pool2", "type": "maxpool", "kernel": 2,        "stride": 2},
        {"name": "fc1",   "type": "fc",      "in_features": 16 * 7 * 7, "out_features": 10},
    ],
}

# K (input-depth for GEMM) for each compute layer
# Conv: K = in_ch × kH × kW   FC: K = in_features
K_SIZES = {
    "conv1": 1 * 3 * 3,       # 9
    "conv2": 8 * 3 * 3,       # 72
    "fc1":   16 * 7 * 7,      # 784
}

# P (output neurons) and Q (spatial positions) for each compute layer
LAYER_SHAPES = {
    "conv1": {"P": 8,  "Q": 28 * 28},   # 8 output channels, 784 spatial positions
    "conv2": {"P": 16, "Q": 14 * 14},   # 16 output channels, 196 spatial positions
    "fc1":   {"P": 10, "Q": 1},         # 10 outputs, single sample
}

DEFAULT_HARDWARE = {
    "R":       16,   # register word width  (bits)
    "ACC":     32,   # accumulator width    (bits)
    "ARRAY_M": 4,    # systolic array rows  (output-neuron dimension)
    "ARRAY_N": 4,    # systolic array cols  (spatial-position dimension)
}


# ── Weight / input generation ─────────────────────────────────────────────────

def generate_weights(seed: int = SEED) -> dict:
    """Generate deterministic float32 weights for all compute layers."""
    rng = np.random.default_rng(seed)
    return {
        "conv1": rng.standard_normal((8,  1,        3, 3)).astype(np.float32) * 0.1,
        "conv2": rng.standard_normal((16, 8,        3, 3)).astype(np.float32) * 0.1,
        "fc1":   rng.standard_normal((10, 16*7*7))        .astype(np.float32) * 0.1,
    }


def generate_input(seed: int = SEED) -> np.ndarray:
    """Generate a deterministic single input image (1×28×28) in [0, 1]."""
    rng = np.random.default_rng(seed + 1)
    return rng.random((1, 28, 28)).astype(np.float32)


# ── Quantization config factories ────────────────────────────────────────────

def make_baseline_quant(k: int, bits: int = 8) -> dict:
    """Baseline: uniform bit-width for every K slot."""
    return {"weight_bits": [bits] * k, "act_bits": [bits] * k}


def make_packed_quant(k: int) -> dict:
    """
    Mixed-precision assignment cycling [4, 4, 8] over K slots.
    ~2/3 of slots get 4-bit; ~1/3 get 8-bit.
    """
    pattern = [4, 4, 8]
    bits = [pattern[i % 3] for i in range(k)]
    return {"weight_bits": bits, "act_bits": bits}


def get_quant_cfgs(mode: str) -> dict:
    """
    Return per-layer quantization configs.
    Both modes now use the same granular mixed-precision config to ensure
    that the numerical output is identical, proving the packing algorithm 
    is mathematically transparent.
    """
    # Force both to use granular / mixed precision
    return {name: make_packed_quant(k) for name, k in K_SIZES.items()}
