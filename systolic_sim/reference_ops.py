"""
reference_ops.py

Plain NumPy reference implementations of conv2d, relu, maxpool, and linear.
These are float32 ops used as the ground truth for correctness validation.
Also includes symmetric signed quantization helpers.
"""
import numpy as np
from typing import Tuple


# ── Quantization helpers ──────────────────────────────────────────────────────

def compute_scale(x: np.ndarray, bits: int) -> float:
    """Per-tensor symmetric quantization scale."""
    amax = float(np.max(np.abs(x)))
    qmax = 2 ** (bits - 1) - 1
    return amax / qmax if amax > 0 else 1.0


def quantize_tensor(x: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    """
    Symmetric signed quantization.
    Returns (int32 quantized array, float scale factor).
    """
    scale = compute_scale(x, bits)
    qmax  = 2 ** (bits - 1) - 1
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32)
    return q, scale


def dequantize_tensor(q: np.ndarray, scale: float) -> np.ndarray:
    """Recover float32 from integer quantized array."""
    return q.astype(np.float32) * scale


# ── im2col ────────────────────────────────────────────────────────────────────

def im2col(x: np.ndarray, kH: int, kW: int, stride: int = 1) -> np.ndarray:
    """
    im2col: reshape input feature map into a column matrix for GEMM convolution.
    x:       (C_in, H_padded, W_padded)
    Returns: (C_in*kH*kW,  H_out*W_out)
    Each column is one flattened kernel window.
    """
    C, H, W = x.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    col   = np.zeros((C * kH * kW, H_out * W_out), dtype=x.dtype)
    for ho in range(H_out):
        for wo in range(W_out):
            patch = x[:, ho*stride:ho*stride+kH, wo*stride:wo*stride+kW]
            col[:, ho * W_out + wo] = patch.reshape(-1)
    return col


# ── Core reference ops ────────────────────────────────────────────────────────

def ref_conv2d(x: np.ndarray, weight: np.ndarray,
               stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Reference conv2d using im2col + GEMM.
    x:      (C_in, H, W)
    weight: (C_out, C_in, kH, kW)
    out:    (C_out, H_out, W_out)
    """
    C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    col   = im2col(x, kH, kW, stride)                        # (K, Q)
    W_mat = weight.reshape(C_out, -1)                         # (P, K)
    H_out = (x.shape[1] - kH) // stride + 1
    W_out = (x.shape[2] - kW) // stride + 1
    return (W_mat @ col).reshape(C_out, H_out, W_out)


def ref_relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU."""
    return np.maximum(0.0, x)


def ref_maxpool2d(x: np.ndarray, kernel: int = 2, stride: int = 2) -> np.ndarray:
    """
    2-D max pooling.
    x:   (C, H, W)
    out: (C, H_out, W_out)
    """
    C, H, W = x.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1
    out   = np.empty((C, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            out[:, i, j] = (
                x[:, i*stride:i*stride+kernel, j*stride:j*stride+kernel]
                .reshape(C, -1).max(axis=1)
            )
    return out


def ref_linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Fully connected layer.
    x:      (in_features,)
    weight: (out_features, in_features)
    out:    (out_features,)
    """
    return weight @ x


# ── Dynamic Reference Runner (PyTorch-based) ──────────────────────────────────

def run_reference_network(model, inp_tensor: np.ndarray) -> dict:
    """
    Run the PyTorch model and capture activations for all Conv and Linear layers.
    model:      The instantiated PyTorch nn.Module.
    inp_tensor: NumPy array (C, H, W).
    """
    import torch
    acts = {"inputs": {}, "outputs": {}}
    
    def get_hook(name):
        def hook(model, input, output):
            # Capture input and output as NumPy
            # input is a tuple (x,)
            acts["inputs"][name] = input[0].detach().cpu().numpy()[0]
            acts["outputs"][name] = output.detach().cpu().numpy()[0]
        return hook


    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Prepare input
    x = torch.from_numpy(inp_tensor).unsqueeze(0).float()
    
    # Run forward
    model.eval()
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    return acts

