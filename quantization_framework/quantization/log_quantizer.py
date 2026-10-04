"""
Log2 quantizer for post-softmax attention weights.

Reference: RepQ-ViT (Li et al., ICCV 2023, arXiv:2212.08254) — log2/log-sqrt-2
quantization for post-softmax activations. AdaLog (ECCV 2024, arXiv:2407.12951)
generalizes the log base via calibration-time search.

Why post-softmax activations need a different quantizer than the rest of the
network: softmax output follows a power-law distribution — most probability
mass sits near 0 (most attention weights are near-irrelevant), with a small
number of large values near the head's peak (the few tokens actually
attended to). A uniform (equal-width) quantization grid, as used elsewhere
in this codebase via PerChannelActivationQuantizer, wastes almost all of its
levels on the near-empty region.

Implementation history (two design iterations corrected via a standalone
synthetic-data test before any model was touched — see
test_log_quantizer_standalone.py):

1. First version fixed scale s=1.0 for every head. Wrong: a "diffuse" head's
   real peak can be well below 1, wasting most of the grid on an unreached
   region. Fixed with per-head calibration.

2. Second version calibrated scale via EMA of each head's running max, and
   CLAMPED any value whose true magnitude fell below the deepest grid level
   to that level's (nonzero) value. This lost badly to a plain per-head
   adaptive linear quantizer (up to 200x worse MSE at 2-bit) across a range
   of synthetic softmax sharpness levels. Root-caused by reading RepQ-ViT's
   and AdaLog's actual public implementations directly (not just their
   papers): both (a) ZERO-FLUSH values that overflow the grid's depth
   instead of clamping them to the deepest nonzero level, and (b) calibrate
   scale via brute-force search over a few percentile candidates, picking
   whichever minimizes actual reconstruction MSE on calibration data — not
   a running max.

   Why zero-flush matters: with ~197 attention positions summing to 1, most
   individual values are near 0 (e.g. ~0.0005). At 2-bit, the deepest grid
   level (index 3) sits at scale/8 — still a real, comparatively large
   number (e.g. ~0.11 if scale~0.9). CLAMPING every near-zero value up to
   that level creates a massive systematic bias (true ~0.0005 vs
   reconstructed ~0.11). ZEROING those same values instead (since they
   overflow the grid's representable depth) reconstructs them near-exactly.
   This is the dominant fix, confirmed by two independently-authored
   reference implementations agreeing on the same behavior.

Per-head vs per-tensor calibration: the reference implementations calibrate
ONE scale per whole attention tensor (per module), not per head — they
don't do per-head bit-width assignment at all. This codebase's mixed-
precision search assigns bit-widths per channel/head, so calibration is
kept PER HEAD here (each head gets its own percentile-searched scale) —
adapting the reference's percentile-search + zero-flush mechanism (the
actual fix) to this codebase's per-channel architecture, not copying their
per-tensor granularity choice (which is orthogonal to the fix and doesn't
fit how the rest of this codebase's search reasons about bit assignment).

Quantize:   x_raw   = round(-log2(clamp(x/delta, min=eps)))
            in_range = x_raw < n_levels          (n_levels = 2^bits)
            x_idx   = clamp(x_raw, 0, n_levels - 1)
Dequantize: x_dq    = delta * 2^(-x_idx) * in_range     (zero where NOT in_range)

delta is chosen per head via percentile search: try candidate percentiles
of the head's calibration-data abs-value distribution as delta, quantize/
dequantize that same calibration data at each candidate, keep whichever
delta gives lowest reconstruction MSE for that head.
"""

import torch
import torch.nn as nn


class LogSoftmaxQuantizer(nn.Module):
    """
    Per-channel log2 quantizer for post-softmax attention weights.

    Channel granularity is per attention HEAD, not per query position —
    attention weight tensors have shape (B, num_heads, N, N) (or
    (B*num_windows, num_heads, N, N) for windowed attention like Swin); heads
    are the closest structural analog to the "output channel" concept used
    everywhere else in this codebase's per-channel quantization. Per-query-
    row granularity (N can be ~197 for ViT, or per-window for Swin) would be
    far too fine-grained and has no corresponding concept in the existing
    bit-width search.

    Args:
        channel_bits: list of int, one bit-width per attention head.
        head_dim:     index of the head dimension in the forward tensor.
                      Default 1, matching (B, num_heads, N, N).
        eps:          numerical floor before dividing/taking log2, avoiding
                      log(0) and division by zero.
        percentile_candidates: percentiles of each head's abs-value
                      distribution tried as the calibration scale delta.
                      Matches RepQ-ViT's/AdaLog's actual public
                      implementations (confirmed by reading their source).
    """

    def __init__(self, channel_bits: list, head_dim: int = 1, eps: float = 1e-6,
                 percentile_candidates=(0.999, 0.9999, 0.99999)):
        super().__init__()
        self.channel_bits = channel_bits
        self.head_dim = head_dim
        self.eps = eps
        self.percentile_candidates = percentile_candidates
        self.num_heads = len(channel_bits)
        self.calibrated = False

        n_levels = [2 ** b for b in channel_bits]
        self.register_buffer('n_levels', torch.tensor(n_levels, dtype=torch.float32))
        self.register_buffer('delta', torch.ones(self.num_heads))

    def _quantize_dequantize(self, x, delta_view, n_levels_view):
        x_f32 = x.float()
        ratio = (x_f32 / delta_view).clamp(min=self.eps)
        x_raw = torch.round(-torch.log2(ratio))
        in_range = (x_raw < n_levels_view).float()
        x_idx = torch.minimum(torch.maximum(x_raw, torch.zeros_like(x_raw)),
                              n_levels_view - 1)
        x_dq = delta_view * torch.pow(2.0, -x_idx) * in_range
        return x_dq

    def calibrate(self, x):
        """
        One-shot per-head percentile-search calibration. See module
        docstring for why this replaces a running-max EMA: percentile
        search + zero-flush is what the actual reference implementations
        do, and a running max alone (with clamp-not-zero-flush) was found
        to lose badly to a plain linear quantizer via a standalone test.
        """
        device = x.device
        self.n_levels = self.n_levels.to(device)
        d = self.head_dim % x.dim()

        view_shape = [1] * x.dim()
        view_shape[d] = self.num_heads
        n_levels_view = self.n_levels.view(view_shape)

        x_perm = x.detach().moveaxis(d, 0).reshape(self.num_heads, -1).float()

        best_delta = torch.zeros(self.num_heads, device=device)
        best_mse = torch.full((self.num_heads,), float('inf'), device=device)

        for p in self.percentile_candidates:
            candidate_delta = torch.quantile(x_perm.abs(), p, dim=1).clamp(min=self.eps)
            delta_view = candidate_delta.view(view_shape)
            x_dq = self._quantize_dequantize(x, delta_view, n_levels_view)

            err = (x_dq.float() - x.float()) ** 2
            err_perm = err.detach().moveaxis(d, 0).reshape(self.num_heads, -1)
            mse_per_head = err_perm.mean(dim=1)

            improve = mse_per_head < best_mse
            best_mse = torch.where(improve, mse_per_head, best_mse)
            best_delta = torch.where(improve, candidate_delta, best_delta)

        self.delta.copy_(best_delta.to(self.delta.device))
        self.calibrated = True

    def forward(self, x):
        device = x.device
        if self.n_levels.device != device:
            self.n_levels = self.n_levels.to(device)
            self.delta = self.delta.to(device)

        d = self.head_dim % x.dim()
        num_heads_actual = x.shape[d]
        if num_heads_actual != self.num_heads:
            raise RuntimeError(
                f'\nLogSoftmaxQuantizer: head count mismatch.\n'
                f'  Expected : {self.num_heads} heads (from channel_bits)\n'
                f'  Got      : {num_heads_actual} at dim {d} of tensor shape {tuple(x.shape)}\n'
                f'  head_dim was set to {self.head_dim}\n'
            )

        if self.training and not self.calibrated:
            self.calibrate(x)

        view_shape = [1] * x.dim()
        view_shape[d] = self.num_heads
        delta_view = self.delta.view(view_shape)
        n_levels_view = self.n_levels.view(view_shape)

        x_dq = self._quantize_dequantize(x, delta_view, n_levels_view)
        return x_dq.to(x.dtype)

    def extra_repr(self):
        bits_summary = {}
        for b in self.channel_bits:
            bits_summary[b] = bits_summary.get(b, 0) + 1
        return f"heads={self.num_heads}, head_dim={self.head_dim}, bits_dist={bits_summary}"
