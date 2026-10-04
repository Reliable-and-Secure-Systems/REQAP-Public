"""
Standalone correctness test for LogSoftmaxQuantizer — no model, no GPU,
no ImageNet needed. Run this BEFORE touching any real model, per the
staged-validation approach (this session previously caught a fatal
implementation gap in a different technique this same way, before it
wasted a full pipeline run).

Checks:
  1. Basic sanity: max value (x=1) reconstructs exactly; no NaN/Inf,
     including at the near-zero tail and at masked-position zeros.
  2. The actual premise: on a power-law-shaped distribution (mimicking
     real softmax output), does the log2 quantizer achieve lower MSE than
     a plain linear (equal-width) quantizer at the same bit-width? This is
     the entire justification for using a log2 grid here — if it doesn't
     win on synthetic data built to match softmax's known shape, there's
     no reason to expect it to help on a real model, and this stops here.

Usage:
    python test_log_quantizer_standalone.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_quantizer import LogSoftmaxQuantizer


def linear_quantize_dequantize(x, bits, dim=1):
    """
    Minimal per-channel linear (equal-width) quantizer for comparison —
    same math as PerChannelActivationQuantizer's asymmetric scheme
    (activations.py), reimplemented standalone here to avoid needing a
    full model/hook setup for this synthetic-data-only test.
    """
    q_max = 2 ** bits - 1
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    scale = (x_max - x_min) / q_max
    scale = scale.clamp(min=1e-8)
    zp = (-x_min / scale).round().clamp(0, q_max)
    x_int = (x / scale + zp).round().clamp(0, q_max)
    return (x_int - zp) * scale


def make_synthetic_softmax_data(num_heads=8, seq_len=197, temperature=5.0, seed=42):
    """
    Mimics real post-softmax attention weight shape and distribution:
    (num_heads, seq_len) — one softmax distribution per head. Higher
    temperature -> sharper peaks -> more power-law-like.

    NOTE: temperature=5.0 with random Gaussian logits produces a fairly
    mild peak (max ~0.9) — real trained attention heads are often much
    sharper (near-one-hot for heads doing precise positional/token
    alignment). Test across a RANGE of temperatures, not just one, since
    the "mild" case may not be representative of what the model actually
    produces, and a technique that only wins on extreme distributions
    would give a false negative here if only tested mildly.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_heads, seq_len, generator=g) * temperature
    return torch.softmax(logits, dim=-1)


def run_checks_for_temperature(temperature):
    x = make_synthetic_softmax_data(temperature=temperature)
    print(f'\n--- temperature={temperature} ---')
    print(f'Synthetic softmax-like data: shape {tuple(x.shape)}, '
          f'range [{x.min().item():.6f}, {x.max().item():.6f}]')

    # ---- Check 1: basic sanity ----
    bits = 4
    q = LogSoftmaxQuantizer(channel_bits=[bits] * x.shape[0], head_dim=0)
    x_dq = q(x)

    has_nan = torch.isnan(x_dq).any().item()
    has_inf = torch.isinf(x_dq).any().item()
    if has_nan or has_inf:
        print(f'  *** FAIL: NaN={has_nan} Inf={has_inf} in output. Stop.')
        return None

    # With per-head adaptive scale, the calibrated scale IS the batch max
    # on first call (self.initialized was False), so the max value should
    # now reconstruct near-exactly (s = batch_max -> x/s = 1.0 at the max
    # -> x_q = 0 -> x_dq = s = original max). This is a direct check that
    # the adaptive-scale fix actually took effect, not just a repeat of
    # the old (incorrect) fixed-scale-at-1.0 assumption.
    max_per_head = x.max(dim=-1).values
    max_dq_per_head = x_dq.max(dim=-1).values
    max_diff = (max_per_head - max_dq_per_head).abs().max().item()
    print(f'  Max value reconstruction: max abs diff = {max_diff:.6f} (should be ~0 now that scale is calibrated per-head, not fixed at 1.0)')

    x_with_zeros = x.clone()
    x_with_zeros[0, :50] = 0.0
    x_dq_zeros = q(x_with_zeros)
    zero_nan = torch.isnan(x_dq_zeros).any().item()
    zero_inf = torch.isinf(x_dq_zeros).any().item()
    if zero_nan or zero_inf:
        print(f'  *** FAIL: zero-input handling broken (NaN={zero_nan}, Inf={zero_inf}). Stop.')
        return None
    print(f'  Zero-input handling: OK (NaN={zero_nan}, Inf={zero_inf})')

    # ---- Check 2: does log2 actually win on this distribution shape? ----
    print(f'  {"Bits":>6} {"Log2 MSE":>14} {"Linear MSE":>14} {"Log2 wins":>12}')
    results = []
    for bits in [2, 3, 4, 5, 6, 8]:
        q_log = LogSoftmaxQuantizer(channel_bits=[bits] * x.shape[0], head_dim=0)
        x_dq_log = q_log(x)
        mse_log = ((x_dq_log - x) ** 2).mean().item()

        x_dq_lin = linear_quantize_dequantize(x, bits, dim=1)
        mse_lin = ((x_dq_lin - x) ** 2).mean().item()

        wins = mse_log < mse_lin
        results.append(wins)
        print(f'  {bits:>6} {mse_log:>14.8f} {mse_lin:>14.8f} {"YES" if wins else "NO":>12}')

    return results


def main():
    print('=' * 70)
    print('STANDALONE LOG2 QUANTIZER CORRECTNESS TEST')
    print('=' * 70)
    print('(mimics real post-softmax attention weights: 8 heads, 197 positions')
    print(' each, each row sums to 1 — testing across a RANGE of sharpness levels,')
    print(' since one mild-temperature sample earlier gave a misleading picture)')

    # Range from mild (previous test's temperature) to very sharp
    # (near-one-hot, closer to what precisely-aligned trained attention
    # heads often produce).
    temperatures = [5.0, 10.0, 20.0, 40.0]
    all_results = {}
    for t in temperatures:
        r = run_checks_for_temperature(t)
        if r is None:
            print('\n*** Stopping — a correctness check failed. Fix before proceeding.')
            return
        all_results[t] = r

    print('\n' + '=' * 70)
    print('SUMMARY across sharpness levels')
    print('=' * 70)
    bits_list = [2, 3, 4, 5, 6, 8]
    header = '  '.join(f'{b}-bit' for b in bits_list)
    print(f'{"Temp":>8}   {header}')
    for t, r in all_results.items():
        row = '  '.join(f'{"WIN" if w else "lose":>6}' for w in r)
        print(f'{t:>8}   {row}')

    win_rate = sum(sum(r) for r in all_results.values()) / (len(all_results) * len(bits_list))
    print(f'\nOverall win rate: {win_rate:.1%}')
    if win_rate > 0.7:
        print('--> Log2 quantizer wins on most configurations, especially at sharper')
        print('    (more realistic) distributions. Worth proceeding to model wiring.')
    elif win_rate > 0.3:
        print('--> Mixed results — wins at some sharpness levels/bit-widths, not others.')
        print('    Worth checking whether real model attention distributions land in')
        print('    the winning regime before investing in model wiring.')
    else:
        print('--> Log2 quantizer still loses on most configurations even at sharper')
        print('    distributions. The adaptive-scale fix alone was not enough — do not')
        print('    proceed to model wiring without further investigation.')
    print('=' * 70)


if __name__ == '__main__':
    main()
