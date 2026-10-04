"""
Baseline Sensitivity Metrics for Comparison
============================================
Implements alternative sensitivity metrics from SOTA mixed-precision papers
to compare against our RMS gradient norm approach.

Metrics:
  --metric hawqv2    HAWQ-V2: average Hessian trace per layer = Tr(H_L)/n_params_L,
                     broadcast to all filters equally. (Dong et al., NeurIPS 2020)
                     arXiv: 1911.03852  |  m=50 Hutchinson vectors (paper default)

  --metric gradnorm  First-order gradient norm per layer = E[||∂L/∂a_i||_F],
                     broadcast to all filters equally. (Chauhan et al., ICCVW 2023)
                     https://openaccess.thecvf.com/content/ICCV2023W/RCV/...
                     Cheaper than Hessian — single backward pass per batch.

  --metric fisher    Diagonal Fisher per filter = (1/N) Σ ||∂L/∂W_c||²_F.
                     Squared weight gradients accumulated per output filter.
                     Per-filter granularity. (SqueezeLLM diagonal Fisher concept,
                     ICML 2023, arXiv: 2306.07629)

  --metric mse       MSE-based gradient-free sensitivity per filter.
                     S_c = ||output_c(W) - output_c(Q(W))||² / ||output_c(W)||²
                     No backpropagation needed — forward pass only.
                     Per-filter granularity.

  --metric l2grad    Pre-Fix2 ablation: identical to our method (weight error for
                     Conv2d, (weight_error + act_error) × gradient_norm for Linear)
                     but using L2 norm instead of RMS for the gradient component.
                     Runs channel_sensitivity.py with --gradient-norm l2.

  --metric adabm     AdaBM: average std of output activations per layer.
                     s^k = E[std(activations_k)] over calibration batches.
                     Forward pass only — no backprop. (Hong et al., CVPR 2024, arXiv: 2404.03296)

  --metric sensiboost  SensiBoost: normalized activation MSE per layer.
                     s_i = ||W_i·X − Q(W_i)·X||²₂ / ||W_i·X||₁
                     Squared activation error normalized by activation L1 magnitude.
                     Forward pass only. (Zhang et al., arXiv: 2503.06518, 2025)

  --metric taylor    First-order Taylor sensitivity per filter = ||W_c ⊙ ∂L/∂W_c||_F
                     Element-wise product of weights and their gradients, summed per
                     output filter. Approximates the loss change from zeroing out filter c.
                     Per-filter granularity. (arXiv 2505.13060, 2025)

  --metric hmqat     HMQAT approximation (Huang et al., Neural Networks 2025).
                     WARNING: paper is paywalled, exact formula unverified.
                     Uses average Hessian trace (same as HAWQ-V2) as best guess.
                     Prefer hawqv2 for any verified comparison.

All metrics produce a CSV in the same format as channel_sensitivity.py:
    layer, channel_idx, sensitivity  (normalized [0,1])

Fed directly into joint_granular_search.py unchanged.

Usage:
    python baseline_sensitivity_metrics.py \\
        --model vgg11_bn --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 --metric hawqv2

    python baseline_sensitivity_metrics.py \\
        --model vgg11_bn --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 --metric gradnorm

    python baseline_sensitivity_metrics.py \\
        --model vgg11_bn --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 --metric fisher

    python baseline_sensitivity_metrics.py \\
        --model vgg11_bn --checkpoint models/vgg11_bn.pt \\
        --dataset cifar10 --metric mse
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pipeline import (
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader,
)
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor


# ---------------------------------------------------------------------------
# Data loader helper
# ---------------------------------------------------------------------------

def get_dataloader(dataset, train=False, batch_size=64, input_size=None):
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
# Hutchinson estimator — per-layer Hessian trace (HAWQ-V2 / HMQAT)
# ---------------------------------------------------------------------------

def compute_hessian_trace_per_layer(model, dataloader, device,
                                     n_batches=10, n_vectors=50):
    """
    Estimate the Hessian trace for each Conv2d and Linear layer using the
    Hutchinson estimator, computed INDEPENDENTLY per layer:

        trace(H_L) ≈ (1/V) * Σ_v [ z_v^T * H_LL * z_v ]

    H_LL * z = d(g_L · z) / dθ_L  (grad-of-grad, layer L only).

    HAWQ-V2 paper (NeurIPS 2020, arXiv 1911.03852) uses m=50 Rademacher vectors
    over 512 calibration samples. cross-entropy loss.

    NOTE: g_dot_z computed PER LAYER in isolation — summing across layers before
    differentiating introduces off-diagonal cross-layer Hessian terms.

    Returns:
        dict: {layer_name: raw_trace (float)}
    """
    layer_names = []
    layer_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
            layer_params[name] = module.weight

    trace_accum = {name: 0.0 for name in layer_names}
    count = 0
    criterion = nn.CrossEntropyLoss()

    for m in model.modules():
        if isinstance(m, nn.ReLU) and m.inplace:
            m.inplace = False

    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        grads = torch.autograd.grad(
            loss, [layer_params[n] for n in layer_names],
            create_graph=True, allow_unused=True
        )
        grad_dict = {n: g for n, g in zip(layer_names, grads) if g is not None}

        for name in layer_names:
            g_L = grad_dict.get(name)
            if g_L is None:
                continue
            param_L = layer_params[name]

            for _ in range(n_vectors):
                z = torch.randint_like(param_L, 0, 2).float() * 2 - 1
                g_dot_z = (g_L * z).sum()
                hv = torch.autograd.grad(
                    g_dot_z, param_L,
                    retain_graph=True, allow_unused=True
                )[0]
                if hv is not None:
                    trace_accum[name] += (z * hv).sum().item()

        count += 1
        model.zero_grad()
        print(f"  Hessian batch {batch_idx + 1}/{n_batches} ...", end='\r')

    model.eval()
    print()
    denom = max(count * n_vectors, 1)
    return {name: trace_accum[name] / denom for name in layer_names}


# ---------------------------------------------------------------------------
# HAWQ-V2: average Hessian trace per layer (Dong et al., NeurIPS 2020)
# ---------------------------------------------------------------------------

def compute_hawqv2_sensitivity(model, dataloader, device,
                                n_batches=10, n_vectors=50):
    """
    HAWQ-V2 sensitivity = Tr(H_L) / n_params_L  (average Hessian trace).

    Verified from paper PDF (arXiv 1911.03852, Lemma 1, Eq. 4):
        S_i = (1/n_i) * Tr(∇²_{W_i} L)

    Layer-level: all filters share the same score → global normalization
    to preserve cross-layer ranking (per-layer norm collapses all to 1.0).
    """
    print(f"  [HAWQ-V2] Hessian trace: {n_batches} batches, "
          f"{n_vectors} Hutchinson vectors...")
    traces = compute_hessian_trace_per_layer(
        model, dataloader, device, n_batches, n_vectors
    )

    raw = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in traces:
            n_filters = module.weight.shape[0]
            n_params  = module.weight.numel()
            avg_trace = abs(traces[name]) / max(n_params, 1)
            raw[name] = [avg_trace] * n_filters

    return _normalize_global(raw)


# ---------------------------------------------------------------------------
# HMQAT approximation (Huang et al., Neural Networks 2025)
# ---------------------------------------------------------------------------

def compute_hmqat_sensitivity(model, dataloader, device,
                               n_batches=10, n_vectors=50):
    """
    HMQAT sensitivity approximation.

    WARNING: Paper is behind Elsevier paywall, no arXiv preprint.
    Abstract says "joint average Hessian trace and parameter size."
    Secondary sources suggest parameter size enters as a separate Pareto
    axis in bit selection, NOT multiplied into the score. Most defensible
    interpretation: same metric as HAWQ-V2 (Tr(H)/n_params).

    Prefer --metric hawqv2 for verified formula. This is kept only as
    a reference to show HMQAT is likely equivalent to HAWQ-V2 in metric.
    """
    print(f"  [HMQAT] Formula unverified (paywalled). Using HAWQ-V2 average "
          f"Hessian trace as best approximation.")
    return compute_hawqv2_sensitivity(model, dataloader, device,
                                      n_batches, n_vectors)


# ---------------------------------------------------------------------------
# First-order gradient norm per layer (Chauhan et al., ICCVW 2023)
# ---------------------------------------------------------------------------

def compute_gradnorm_sensitivity(model, dataloader, device, n_batches=10):
    """
    First-order gradient norm sensitivity per layer.

        S_i = E[ ||∂L/∂a_i||_F ]

    where a_i is the output activation of layer i and ||·||_F is the
    Frobenius norm over all (batch, spatial, channel) dimensions.

    Source: Chauhan et al., ICCVW 2023
    "Post Training Mixed Precision Quantization of Neural Networks
    Using First-Order Sensitivity"
    https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/
    Chauhan_Post_Training_Mixed_Precision_Quantization_of_Neural_Networks
    _Using_First-Order_ICCVW_2023_paper.html

    Layer-level: one score per layer, all filters same → global normalization.
    Much cheaper than Hessian: one backward pass per batch, no grad-of-grad.
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    accum = {name: 0.0 for name in layer_names}
    counts = {name: 0 for name in layer_names}
    hooks = []

    def make_hook(name):
        def hook(mod, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            accum[name] += g.detach().norm().item()   # Frobenius norm over all dims
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_full_backward_hook(make_hook(name))
            hooks.append(h)

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

    raw = {}
    for name, module in model.named_modules():
        if name in layer_names and counts[name] > 0:
            n_filters = module.weight.shape[0]
            score = accum[name] / counts[name]
            raw[name] = [score] * n_filters   # same score for all filters

    return _normalize_global(raw)


# ---------------------------------------------------------------------------
# Diagonal Fisher per filter (SqueezeLLM, ICML 2023)
# ---------------------------------------------------------------------------

def compute_fisher_sensitivity(model, dataloader, device, n_batches=10):
    """
    Diagonal Fisher Information per output filter.

        s_c = (1/N) Σ_n  ||∂L_n / ∂W_c||²_F

    For each output channel c, accumulates the sum of squared weight
    gradients over all parameters of that filter (C_in, kH, kW for Conv2d
    or C_in for Linear). Equivalent to the diagonal of the empirical Fisher
    Information Matrix aggregated per output filter.

    Source: SqueezeLLM (ICML 2023, arXiv 2306.07629) diagonal Fisher concept.
    The per-weight version s_ij = E[(∂L/∂w_ij)²] is aggregated here per filter
    for compatibility with our filter-level greedy search.

    Per-filter granularity → per-layer normalization.
    No create_graph needed — uses standard .grad accumulation.
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    accum = {name: None for name in layer_names}
    counts = {name: 0 for name in layer_names}

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

        # Collect squared weight gradients per output filter
        for name, module in model.named_modules():
            if name not in layer_names:
                continue
            if module.weight.grad is None:
                continue
            g = module.weight.grad.detach()            # (C_out, ...)
            n_filters = g.shape[0]
            # Sum squared gradients over all dims except output channel
            per_filter = g.pow(2).reshape(n_filters, -1).sum(dim=1)  # (C_out,)
            if accum[name] is None:
                accum[name] = per_filter.cpu()
            else:
                accum[name] += per_filter.cpu()
            counts[name] += 1

    model.eval()

    raw = {}
    for name in layer_names:
        if accum[name] is not None and counts[name] > 0:
            raw[name] = (accum[name] / counts[name]).tolist()

    return _normalize_per_layer(raw)


# ---------------------------------------------------------------------------
# MSE-based gradient-free sensitivity per filter
# ---------------------------------------------------------------------------

def compute_mse_sensitivity(model, dataloader, device, min_bits=2, n_batches=10):
    """
    MSE-based gradient-free sensitivity per output filter.

        S_c = ||output_c(W) - output_c(Q(W))||² / max(||output_c(W)||², ε)

    For each output filter c, measures the relative change in that filter's
    output when its weights are quantized to min_bits. No backpropagation —
    forward pass only.

    Implementation:
      1. Register forward hooks to capture layer inputs during a forward pass.
      2. For each layer, quantize weights per-filter at min_bits using our
         existing quantize_tensor, then manually recompute the layer output
         using F.conv2d / F.linear.
      3. Compute per-filter squared error relative to original output.

    Per-filter granularity → per-layer normalization.
    """
    layer_names = []
    layer_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
            layer_modules[name] = module

    accum_err  = {name: None for name in layer_names}
    accum_norm = {name: None for name in layer_names}
    counts     = {name: 0 for name in layer_names}
    hooks      = []

    def make_hook(name):
        def hook(module, inp, output):
            x     = inp[0].detach()
            w     = module.weight.detach()
            b     = module.bias.detach() if module.bias is not None else None
            n_filters = w.shape[0]

            # Quantize each output filter independently at min_bits
            w_q = torch.zeros_like(w)
            for c in range(n_filters):
                w_c = w[c].unsqueeze(0)
                q_c, _, _ = quantize_tensor(w_c, bit_width=min_bits,
                                            method='symmetric')
                w_q[c] = q_c.squeeze(0)

            # Recompute output with quantized weights
            if isinstance(module, nn.Conv2d):
                y_orig  = output.detach()                          # (B, C_out, H, W)
                y_quant = F.conv2d(x, w_q, b,
                                   module.stride, module.padding,
                                   module.dilation, module.groups) # (B, C_out, H, W)
                # Per-filter: sum over (B, H, W)
                err  = (y_orig - y_quant).pow(2).sum(dim=(0, 2, 3))  # (C_out,)
                nrm  = y_orig.pow(2).sum(dim=(0, 2, 3))              # (C_out,)
            else:  # Linear
                y_orig  = output.detach()                          # (B, ..., C_out)
                y_quant = F.linear(x, w_q, b)                     # (B, ..., C_out)
                # channels at last dim
                err  = (y_orig - y_quant).pow(2).reshape(-1, n_filters).sum(dim=0)
                nrm  = y_orig.pow(2).reshape(-1, n_filters).sum(dim=0)

            if accum_err[name] is None:
                accum_err[name]  = err.cpu()
                accum_norm[name] = nrm.cpu()
            else:
                accum_err[name]  += err.cpu()
                accum_norm[name] += nrm.cpu()
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    raw = {}
    for name in layer_names:
        if accum_err[name] is not None and counts[name] > 0:
            ratio = accum_err[name] / torch.clamp(accum_norm[name], min=1e-8)
            raw[name] = ratio.tolist()

    return _normalize_per_layer(raw)


# ---------------------------------------------------------------------------
# L2 gradient norm (pre-Fix2 ablation)
# ---------------------------------------------------------------------------

def compute_l2grad_sensitivity(model, dataloader, bit_choices, device, n_batches=10):
    """
    Pre-Fix2 ablation: identical pipeline to our method but using gradient L2 norm
    instead of RMS for Linear layers.

      Conv2d: weight quantization error (unchanged from our method)
      Linear: (weight_error + act_error) × gradient_L2_norm
              (same as ours but L2 instead of RMS for the gradient)

    Cleanly isolates Fix 2 (spatial-size bias in gradient normalization).
    Calls compute_channel_sensitivity() with gradient_norm='l2'.

    Returns:
        tuple: (sensitivity dict, per_bit_act_errors dict)
    """
    from channel_sensitivity import compute_channel_sensitivity
    return compute_channel_sensitivity(
        model, dataloader, bit_choices, device=device,
        n_calib_batches=n_batches, gradient_norm='l2'
    )


# ---------------------------------------------------------------------------
# First-order Taylor sensitivity per filter (arXiv 2505.13060, 2025)
# ---------------------------------------------------------------------------

def compute_taylor_sensitivity(model, dataloader, device, n_batches=10):
    """
    First-order Taylor sensitivity per output filter.

        s_c = ||W_c ⊙ ∂L/∂W_c||_F

    Element-wise product of each filter's weights and their gradients,
    then Frobenius norm. Approximates the first-order loss change when
    filter c is perturbed (zeroed out).

    Source: "Automatic Mixed Precision via First-Order Taylor Expansion"
    arXiv: 2505.13060 (2025)

    Per-filter granularity → per-layer normalization.
    Cost: one backward pass per batch (same as fisher/gradnorm).
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    accum = {name: None for name in layer_names}
    counts = {name: 0 for name in layer_names}

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

        for name, module in model.named_modules():
            if name not in layer_names:
                continue
            if module.weight.grad is None:
                continue
            w = module.weight.detach()          # (C_out, ...)
            g = module.weight.grad.detach()     # (C_out, ...)
            n_filters = w.shape[0]
            # ||W_c ⊙ ∂L/∂W_c||_F per output filter
            taylor = (w * g).abs().reshape(n_filters, -1).norm(dim=1)  # (C_out,)
            if accum[name] is None:
                accum[name] = taylor.cpu()
            else:
                accum[name] += taylor.cpu()
            counts[name] += 1

    model.eval()

    raw = {}
    for name in layer_names:
        if accum[name] is not None and counts[name] > 0:
            raw[name] = (accum[name] / counts[name]).tolist()

    return _normalize_per_layer(raw)


# ---------------------------------------------------------------------------
# AdaBM: activation standard deviation per layer (CVPR 2024)
# ---------------------------------------------------------------------------

def compute_adabm_sensitivity(model, dataloader, device, n_batches=10):
    """
    AdaBM activation sensitivity per layer.

        s^k = E[ std(activations_k) ]

    Average standard deviation of each layer's output activations over
    calibration batches. Layers with high activation variance are more
    sensitive to quantization.

    Source: "AdaBM: On-the-Fly Adaptive Bit Mapping for Image Super-Resolution"
    Hong et al., CVPR 2024 — arXiv: 2404.03296

    Layer-level: one score per layer, broadcast to all filters → global normalization.
    Cost: forward pass only — no backpropagation needed.
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)

    accum = {name: 0.0 for name in layer_names}
    counts = {name: 0 for name in layer_names}
    hooks = []

    def make_hook(name):
        def hook(module, inp, output):
            out = output.detach()
            # std over all elements (batch, channels, spatial)
            accum[name] += out.float().std().item()
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    raw = {}
    for name, module in model.named_modules():
        if name in layer_names and counts[name] > 0:
            n_filters = module.weight.shape[0]
            score = accum[name] / counts[name]
            raw[name] = [score] * n_filters  # same score for all filters

    return _normalize_global(raw)


# ---------------------------------------------------------------------------
# SensiBoost: normalized activation MSE per layer (arXiv 2503.06518, 2025)
# ---------------------------------------------------------------------------

def compute_sensiboost_sensitivity(model, dataloader, device, min_bits=2, n_batches=10):
    """
    SensiBoost activation sensitivity per layer.

        s_i = ||W_i·X − Q(W_i)·X||²_2 / ||W_i·X||_1

    Squared L2 norm of the activation error (FP32 vs quantized weights) divided
    by the L1 norm of the original activations. Normalizing by activation magnitude
    makes the score scale-invariant across layers of different sizes.

    Source: "Towards Superior Quantization Accuracy: A Layer-sensitive Approach"
    Zhang et al., arXiv: 2503.06518 (March 2025)

    Layer-level: one score per layer, broadcast to all filters → global normalization.
    Cost: forward pass only — no backpropagation needed.
    """
    layer_names = []
    layer_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
            layer_modules[name] = module

    accum_err  = {name: 0.0 for name in layer_names}
    accum_norm = {name: 0.0 for name in layer_names}
    counts     = {name: 0 for name in layer_names}
    hooks      = []

    def make_hook(name):
        def hook(module, inp, output):
            x = inp[0].detach()
            w = module.weight.detach()
            b = module.bias.detach() if module.bias is not None else None

            # Quantize entire weight tensor at min_bits (layer-level, not per-filter)
            w_q, _, _ = quantize_tensor(w, bit_width=min_bits, method='symmetric')

            if isinstance(module, nn.Conv2d):
                y_orig  = output.detach()
                y_quant = F.conv2d(x, w_q, b, module.stride, module.padding,
                                   module.dilation, module.groups)
            else:
                y_orig  = output.detach()
                y_quant = F.linear(x, w_q, b)

            err  = (y_orig - y_quant).float().pow(2).sum().item()
            nrm  = y_orig.float().abs().sum().item()
            accum_err[name]  += err
            accum_norm[name] += max(nrm, 1e-8)
            counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    raw = {}
    for name, module in model.named_modules():
        if name in layer_names and counts[name] > 0:
            n_filters = module.weight.shape[0]
            score = accum_err[name] / accum_norm[name]
            raw[name] = [score] * n_filters  # same score for all filters

    return _normalize_global(raw)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_per_layer(raw_scores):
    """
    Normalize each layer's scores to [0, 1] independently.
    Use when filters within a layer have meaningfully different scores
    (fisher, mse, l2grad).
    """
    sensitivity = {}
    for layer_name, scores in raw_scores.items():
        max_val = max(scores) if scores and max(scores) > 0 else 1.0
        sensitivity[layer_name] = [v / max_val for v in scores]
    return sensitivity


def _normalize_global(raw_scores):
    """
    Normalize all scores to [0, 1] across ALL layers globally.
    Use when all filters in a layer share the same score (hawqv2, gradnorm).
    Per-layer normalization would collapse all layers to 1.0, destroying
    the cross-layer ranking that these metrics are designed to capture.
    """
    all_vals = [v for scores in raw_scores.values() for v in scores]
    global_max = max(all_vals) if all_vals and max(all_vals) > 0 else 1.0
    return {
        layer_name: [v / global_max for v in scores]
        for layer_name, scores in raw_scores.items()
    }


# ---------------------------------------------------------------------------
# CSV output (same format as channel_sensitivity.py)
# ---------------------------------------------------------------------------

def save_sensitivity_csv(sensitivity, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'channel_idx', 'sensitivity'])
        for layer_name, scores in sensitivity.items():
            for c_idx, score in enumerate(scores):
                writer.writerow([layer_name, c_idx, f'{score:.6f}'])
    print(f"\nSaved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METRICS = ['hawqv2', 'gradnorm', 'fisher', 'mse', 'adabm', 'sensiboost', 'taylor', 'l2grad', 'hmqat']

def main():
    parser = argparse.ArgumentParser(
        description='Baseline sensitivity metrics for comparison vs our RMS gradient norm'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['vgg11_bn', 'resnet', 'levit', 'swin'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--metric', type=str, required=True,
                        choices=METRICS,
                        help='hawqv2: avg Hessian trace (NeurIPS 2020) | '
                             'gradnorm: first-order grad norm (ICCVW 2023) | '
                             'fisher: diagonal Fisher per filter (ICML 2023) | '
                             'mse: MSE gradient-free per filter | '
                             'adabm: activation std per layer (CVPR 2024) | '
                             'sensiboost: normalized activation MSE per layer (arXiv 2503.06518, 2025) | '
                             'taylor: first-order Taylor ||W⊙∂L/∂W|| (arXiv 2505.13060, 2025) | '
                             'l2grad: pre-Fix2 ablation (L2 vs RMS) | '
                             'hmqat: unverified, uses HAWQ-V2 formula')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--act-error-output', type=str, default=None,
                        help='Output JSON for per-bit activation errors (l2grad only)')
    parser.add_argument('--calib-batches', type=int, default=10)
    parser.add_argument('--min-bits', type=int, default=2,
                        help='Min bit-width for MSE metric (default: 2)')
    parser.add_argument('--n-vectors', type=int, default=50,
                        help='Hutchinson vectors per layer per batch for hawqv2/hmqat. '
                             'HAWQ-V2 paper uses m=50. Use 10 for faster runs.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = args.output or f'{args.model}_{args.dataset}_{args.metric}_sensitivity.csv'
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)

    print('=' * 65)
    print('BASELINE SENSITIVITY METRIC')
    print('=' * 65)
    print(f'  Model:   {args.model}')
    print(f'  Dataset: {args.dataset}')
    print(f'  Metric:  {args.metric}')
    print(f'  Device:  {device}')
    print('=' * 65)

    t0 = time.time()

    model = load_model(args.model, checkpoint_path=args.checkpoint,
                       num_classes=num_classes)
    model = model.to(device)
    model.eval()

    bs = 32 if args.model in ('levit', 'swin') else 64
    input_size = 224 if args.model in ('levit', 'swin') or args.dataset == 'gtsrb' else 32
    loader = get_dataloader(args.dataset, train=False,
                            batch_size=bs, input_size=input_size)

    bits = [2, 4, 8]
    sensitivity = None

    print(f'\nRunning {args.metric} sensitivity...')

    if args.metric == 'hawqv2':
        sensitivity = compute_hawqv2_sensitivity(
            model, loader, device,
            n_batches=args.calib_batches, n_vectors=args.n_vectors
        )
    elif args.metric == 'hmqat':
        sensitivity = compute_hmqat_sensitivity(
            model, loader, device,
            n_batches=args.calib_batches, n_vectors=args.n_vectors
        )
    elif args.metric == 'gradnorm':
        sensitivity = compute_gradnorm_sensitivity(
            model, loader, device, n_batches=args.calib_batches
        )
    elif args.metric == 'fisher':
        sensitivity = compute_fisher_sensitivity(
            model, loader, device, n_batches=args.calib_batches
        )
    elif args.metric == 'mse':
        sensitivity = compute_mse_sensitivity(
            model, loader, device,
            min_bits=args.min_bits, n_batches=args.calib_batches
        )
    elif args.metric == 'adabm':
        sensitivity = compute_adabm_sensitivity(
            model, loader, device, n_batches=args.calib_batches
        )
    elif args.metric == 'sensiboost':
        sensitivity = compute_sensiboost_sensitivity(
            model, loader, device,
            min_bits=args.min_bits, n_batches=args.calib_batches
        )
    elif args.metric == 'taylor':
        sensitivity = compute_taylor_sensitivity(
            model, loader, device, n_batches=args.calib_batches
        )
    elif args.metric == 'l2grad':
        sensitivity, per_bit_act_errors = compute_l2grad_sensitivity(
            model, loader, bits, device, n_batches=args.calib_batches
        )
        if per_bit_act_errors and args.act_error_output:
            with open(args.act_error_output, 'w') as f:
                json.dump(per_bit_act_errors, f, indent=2)
            print(f'Saved per-bit activation errors: {args.act_error_output}')

    # Summary
    total_layers = len(sensitivity)
    total_channels = sum(len(v) for v in sensitivity.values())
    print(f'\nSummary: {total_layers} layers, {total_channels} total filters')
    for name, scores in sensitivity.items():
        print(f'  {name:<55} {len(scores):5d} ch  '
              f'avg={sum(scores)/len(scores):.3f}  max={max(scores):.3f}')

    save_sensitivity_csv(sensitivity, output_path)

    print(f'\nDone in {time.time() - t0:.1f}s')
    print(f'Output: {output_path}')
    print('\nNext: pass this CSV to joint_granular_search.py with --sensitivity')


if __name__ == '__main__':
    main()
