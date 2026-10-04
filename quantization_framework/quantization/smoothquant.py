"""
SmoothQuant-style activation-to-weight difficulty migration.

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models," ICML 2023 (arXiv:2211.10438).

Problem this addresses: per-channel activation outliers (common in
transformer attention/MLP layers, especially post-GELU and post-attention)
force a per-tensor or coarse activation quantizer to waste most of its
dynamic range on a few outlier channels, making the bulk of values collapse
to near-zero precision. Weights are comparatively easy to quantize (no
severe per-channel outliers). SmoothQuant migrates this difficulty: it
divides each input channel of a Linear layer's activation by a per-channel
scale s, and multiplies the corresponding weight column by the same s.

    Y = X @ W^T  =  (X / s) @ (W * s)^T

This is mathematically EXACT in FP32 (the transform is undone by the matmul
itself — no approximation, no accuracy change until quantization is applied
on top). Only the *distribution* each factor exposes to its quantizer
changes: activations become smoother (less outlier-dominated), weights
absorb some of that dynamic range (weights tolerate this far better than
activations do).

Scale formula (per input channel j):
    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

alpha=0.5 splits the migration evenly (SmoothQuant's default, "well-balanced
point" per the original paper). alpha closer to 1.0 pushes more difficulty
onto weights; alpha closer to 0.0 leaves more difficulty on activations.

Implementation choice: this version does NOT fold the scale into a preceding
LayerNorm's affine gamma (SmoothQuant's zero-runtime-cost trick for LLMs).
Folding requires knowing exactly what precedes each Linear layer and whether
that predecessor is a simple elementwise-affine op (LayerNorm) or something
non-foldable (e.g. GELU before fc2, or the attention mechanism before proj)
— getting that wrong silently breaks correctness. Instead, this applies the
scale via an explicit forward pre-hook that divides the layer's input by s
at every forward call. This costs one extra elementwise op per Linear layer
at inference (irrelevant for PTQ accuracy evaluation; a real hardware
deployment would want the folded version as a separate optimization pass)
but is unconditionally correct regardless of what precedes the layer.

Must be applied BEFORE any weight/activation quantization — it modifies the
FP32 model in place, and downstream quantization then operates on the
already-smoothed weights and (via the hook) already-smoothed activations.
"""

import torch
import torch.nn as nn


def apply_smoothquant(model, calib_loader, device, alpha=0.5,
                       n_calib_batches=10, eps=1e-5, layer_filter=None):
    """
    Compute and apply SmoothQuant per-input-channel scaling to every
    nn.Linear layer in `model` (or a filtered subset). Modifies the model
    in place: weight columns are permanently rescaled, and a persistent
    forward pre-hook is registered on each smoothed layer to rescale its
    input at every subsequent forward call (including inside the returned
    model's normal use for calibration/evaluation afterward).

    Args:
        model:            PyTorch model, already on `device`, eval mode.
        calib_loader:     DataLoader yielding (images, labels) for the
                          calibration forward passes used to measure
                          per-input-channel activation abs-max.
        device:           torch device.
        alpha:            migration strength in [0, 1]. 0.5 = SmoothQuant's
                          documented default, even split.
        n_calib_batches:  number of batches to run for calibration.
        eps:              numerical floor to avoid divide-by-zero when a
                          channel has near-zero activation or weight range.
        layer_filter:     optional callable(name, module) -> bool. If given,
                          only Linear layers for which this returns True are
                          smoothed. Default: all nn.Linear layers.

    Returns:
        model (same object, modified in place)
        stats: dict {layer_name: {'alpha': alpha, 's_min':..., 's_max':...,
               's_mean':...}} — summary of the scale applied per layer, for
               logging/verification. Layers with no calibration signal
               (never triggered during the calibration pass) are omitted.
    """
    target_modules = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if layer_filter is not None and not layer_filter(name, module):
            continue
        target_modules[name] = module

    # ---- Step 1: calibration pass — per-input-channel activation abs-max ----
    act_absmax = {}  # {module: (in_features,) tensor, running max across batches}
    calib_hooks = []

    def make_calib_hook(mod):
        def hook(module, inputs):
            x = inputs[0]
            if x.dim() < 1:
                return
            flat = x.detach().abs().reshape(-1, x.shape[-1])
            batch_max = flat.max(dim=0).values
            if mod not in act_absmax:
                act_absmax[mod] = batch_max
            else:
                act_absmax[mod] = torch.maximum(act_absmax[mod], batch_max)
        return hook

    for module in target_modules.values():
        h = module.register_forward_pre_hook(make_calib_hook(module))
        calib_hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= n_calib_batches:
                break
            model(images.to(device))

    for h in calib_hooks:
        h.remove()

    # ---- Step 2: compute scale, rescale weights, install permanent hook ----
    stats = {}
    for name, module in target_modules.items():
        if module not in act_absmax:
            continue  # layer never fired during calibration (e.g. dead branch)

        a_max = act_absmax[module].to(device)                       # (in_features,)
        w_max = module.weight.detach().abs().max(dim=0).values.to(device)  # (in_features,)

        s = (a_max.clamp(min=eps) ** alpha) / (w_max.clamp(min=eps) ** (1 - alpha))
        s = s.clamp(min=eps)

        if s.shape[0] != module.weight.shape[1]:
            raise RuntimeError(
                f"SmoothQuant scale/weight shape mismatch for '{name}': "
                f"scale has {s.shape[0]} entries, weight expects "
                f"{module.weight.shape[1]} input channels."
            )

        # Permanently rescale weight columns (per input channel). Bias is
        # untouched — it's added after the matmul, unaffected by input-side
        # scaling.
        module.weight.data = module.weight.data * s.unsqueeze(0)

        def make_scale_hook(scale):
            def hook(mod, inputs):
                x = inputs[0]
                return (x / scale,) + inputs[1:]
            return hook

        module.register_forward_pre_hook(make_scale_hook(s))

        stats[name] = {
            'alpha': alpha,
            's_min': s.min().item(),
            's_max': s.max().item(),
            's_mean': s.mean().item(),
        }

    return model, stats
