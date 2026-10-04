"""
ERQ-style closed-form activation-quantization-error compensation ("Aqer").

Reference: ERQ (arXiv:2407.06794) — implementation verified directly against
the actual public repo (github.com/zysxmu/ERQ, test_quant_expand.py,
replace_W function), not the paper text alone.

Mechanism, and why it's different from every calibration-only technique
already tried this session (SmoothQuant, log2, twin-GELU — all of which
picked a smarter FIXED quantization grid/scale): Aqer does not touch the
quantization grid at all. It treats activation-quantization error as noise
and SOLVES a per-layer ridge regression that updates the FP32 weights (and
bias) so that, when the layer is fed the ALREADY-QUANTIZED activation, its
output best approximates what the ORIGINAL FP32 weights would have produced
from the ORIGINAL FP32 (unquantized) activation. It is a direct error-
compensation step, not a distribution-shape-matching heuristic.

Math (verified against the reference source, variable names kept close to
the original for traceability):

  Given, for one Linear/Conv2d layer, calibration-time tensors:
    quan_input:   (N, in_features)   — the layer's ACTUAL input, AFTER the
                  layer's own (already-calibrated) activation quantizer.
    true_output:  (N, out_features)  — the layer's TRUE FP32 output,
                  computed from the ORIGINAL (unquantized) FP32 input.
    W_orig:       (out_features, in_features)
    b_orig:       (out_features,)

  A = [quan_input | 1]                                   (N, in_features+1)
  Y = true_output - A @ [W_orig | b_orig]^T               (N, out_features)
      (residual: how much the quantized input's error, propagated through
      the ORIGINAL weights, deviates from the true FP32 output)
  beta = (A^T A + lambda*I)^-1 A^T Y                      (in_features+1, out_features)
  W_new = W_orig + beta[:-1, :]^T
  b_new = b_orig + beta[-1, :]

  Solved ONCE per layer, jointly across all output channels (one matrix
  solve produces every output channel's correction simultaneously).

Critical property (confirmed from the reference implementation): A^T A only
depends on the CALIBRATED ACTIVATION QUANTIZER (via quan_input), not on any
WEIGHT bit-width. This makes Aqer bit-width-agnostic — it can be computed
ONCE per layer, as a one-shot correction to the FP32 weights, BEFORE the
per-channel weight bit-width search runs. The search then quantizes these
corrected weights per candidate config exactly as before (standard
per-channel round-to-nearest via quantize_tensor) — Aqer does not replace
or interact with that step, it only changes what FP32 weights the search
starts from.

Deliberately NOT implemented here: "Wqer" (ERQ's second, iterative
weight-ROUNDING correction stage) — confirmed from the reference source to
be bit-width-DEPENDENT (calls the weight quantizer internally per
candidate), so it cannot be a one-shot pre-search step the way Aqer is; it
would need to rerun per candidate config inside the search loop, which is
a materially larger integration cost. Deferred as a separate, later
addition per Fable's research (rated higher complexity), not needed to
validate whether the core error-compensation idea helps at all.

Numerical note: the reference implementation uses torch.inverse directly.
This implementation uses torch.linalg.solve instead (numerically more
stable — avoids explicitly forming the matrix inverse, standard practice
for solving a linear system).
"""

import torch


def compute_aqer_correction(W_orig, b_orig, quan_input, true_output, lam=0.1):
    """
    Solve the Aqer ridge regression for one Linear-equivalent layer and
    return the corrected weight and bias. Does not modify inputs in place.

    Args:
        W_orig:      (out_features, in_features) FP32 weight tensor.
        b_orig:      (out_features,) FP32 bias tensor. Pass torch.zeros(...)
                     if the layer has no bias.
        quan_input:  (N, in_features) the layer's actual input AFTER its
                     own calibrated activation quantizer has been applied.
        true_output: (N, out_features) the layer's true FP32 output,
                     computed from the ORIGINAL (unquantized) input.
        lam:         ridge regularization strength. ERQ's reference script
                     hand-tunes this via a CLI constant per bit-width
                     (not cross-validated) — this implementation exposes it
                     as a parameter; callers needing an automatic choice
                     should search a small candidate set on held-out
                     calibration data (see test_erq_correction_standalone.py
                     for the validation methodology this depends on).

    Returns:
        (W_new, b_new): corrected weight and bias tensors, same shapes and
        dtype as W_orig/b_orig.

    Raises:
        RuntimeError if N <= in_features + 1 (the regression is
        underdetermined — need more calibration samples than the number of
        parameters being solved for per output channel).
    """
    device = W_orig.device
    dtype = torch.float64  # solve in double precision for numerical
                            # stability given this session's history of
                            # subtle numerical issues; cast back at the end

    W_orig_f = W_orig.detach().to(dtype)
    b_orig_f = b_orig.detach().to(dtype)
    quan_input_f = quan_input.detach().to(dtype)
    true_output_f = true_output.detach().to(dtype)

    N, in_features = quan_input_f.shape
    out_features = W_orig_f.shape[0]

    if N <= in_features + 1:
        raise RuntimeError(
            f'compute_aqer_correction: underdetermined regression. '
            f'N={N} calibration samples <= in_features+1={in_features + 1} '
            f'parameters per output channel. Need more calibration data '
            f'(more images and/or more tokens per image) before this layer '
            f'can be corrected safely.'
        )

    ones = torch.ones(N, 1, device=device, dtype=dtype)
    A = torch.cat([quan_input_f, ones], dim=1)  # (N, in_features+1)

    W_ext = torch.cat([W_orig_f, b_orig_f.unsqueeze(1)], dim=1)  # (out_features, in_features+1)
    predicted_from_quant = A @ W_ext.T  # (N, out_features)
    Y = true_output_f - predicted_from_quant  # (N, out_features)

    AtA = A.T @ A  # (in_features+1, in_features+1)
    AtY = A.T @ Y  # (in_features+1, out_features)
    reg = lam * torch.eye(in_features + 1, device=device, dtype=dtype)

    beta = torch.linalg.solve(AtA + reg, AtY)  # (in_features+1, out_features)

    beta_W = beta[:-1, :].T  # (out_features, in_features)
    beta_b = beta[-1, :]     # (out_features,)

    W_new = (W_orig_f + beta_W).to(W_orig.dtype)
    b_new = (b_orig_f + beta_b).to(b_orig.dtype)

    return W_new, b_new
