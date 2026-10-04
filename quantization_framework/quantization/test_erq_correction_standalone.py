"""
Standalone correctness test for compute_aqer_correction — no model, no GPU,
no ImageNet needed. Run this BEFORE touching any real model, same staged-
validation approach used for log_quantizer.py and twin_gelu_quantizer.py.

Unlike those two (which compare quantizer SHAPES on a fixed synthetic
distribution), Aqer explicitly SOLVES a regression to minimize error on
calibration data. The right thing to test here is GENERALIZATION: does the
correction reduce error on HELD-OUT data (not seen during the ridge-
regression fit), not just on the calibration set it was fit on. A
correction that only fits the calibration set well but doesn't generalize
would be useless (or actively harmful) once wired into a real model
evaluated on unseen images.

Checks:
  1. Basic sanity: with NO quantization error (quan_input == true FP32
     input exactly), the correction should be near-zero — there's nothing
     to compensate for, so beta should solve close to 0 and W_new ~= W_orig.
     This validates the math/shapes independent of whether quantization
     itself is involved.
  2. The actual premise: with REAL per-channel quantization error injected,
     does the Aqer-corrected (W_new, b_new) produce lower output MSE than
     the original (W_orig, b_orig) on a HELD-OUT test set, using the same
     quantizer (calibrated once on the calibration set, not re-calibrated
     on the test set — matches how a real deployed quantizer works)?

Usage:
    python test_erq_correction_standalone.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erq_correction import compute_aqer_correction


def per_channel_quantize(x, bits, calib_min=None, calib_max=None):
    """
    Simple per-channel (per-input-feature) linear asymmetric quantizer,
    matching the scheme actually in use throughout this codebase
    (PerChannelActivationQuantizer). If calib_min/calib_max are given,
    quantizes x using THOSE calibrated ranges (for the held-out test set,
    reusing calibration-set-derived ranges rather than recalibrating on
    test data, which would leak test-set information).

    Returns (x_quantized, used_min, used_max).
    """
    q_max = 2 ** bits - 1
    if calib_min is None:
        calib_min = x.min(dim=0, keepdim=True).values
        calib_max = x.max(dim=0, keepdim=True).values
    scale = (calib_max - calib_min) / q_max
    scale = scale.clamp(min=1e-8)
    zp = (-calib_min / scale).round().clamp(0, q_max)
    x_int = (x / scale + zp).round().clamp(0, q_max)
    x_dq = (x_int - zp) * scale
    return x_dq, calib_min, calib_max


def make_layer(in_features, out_features, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_features, in_features, generator=g) * (1.0 / in_features ** 0.5)
    b = torch.randn(out_features, generator=g) * 0.1
    return W, b


def make_mixing_matrix(in_features, seed):
    """
    The correlation STRUCTURE shared between calibration and test data —
    generated ONCE and reused, so both splits represent "the same
    distribution, different samples" (as in a real calibrate-then-evaluate
    scenario), not two unrelated distributions. An earlier version of this
    test regenerated the mixing matrix fresh per split (different seeds),
    which silently made calibration and test data come from DIFFERENT
    correlation structures entirely — any regression fit on one would fail
    to generalize to the other regardless of the technique being tested,
    which is not a meaningful test of anything. Caught by the generalization
    check itself failing catastrophically (-2315% at 4-bit) while the
    zero-quantization-error sanity check passed cleanly — pointing at the
    test's data generation, not the correction math.
    """
    g = torch.Generator().manual_seed(seed)
    n_factors = max(4, in_features // 8)
    mixing = torch.randn(n_factors, in_features, generator=g) * 0.5
    return mixing, n_factors


def make_correlated_inputs(n_samples, mixing, n_factors, seed):
    """
    Correlated (not i.i.d.) synthetic input data — closer to real
    activations, which have inter-feature correlation, than pure i.i.d.
    Gaussian noise would be. Uses a SHARED mixing matrix (see
    make_mixing_matrix) so different calls represent different samples
    from the SAME distribution, not different distributions.
    """
    g = torch.Generator().manual_seed(seed)
    factors = torch.randn(n_samples, n_factors, generator=g)
    x = factors @ mixing + torch.randn(n_samples, mixing.shape[1], generator=g) * 0.3
    return x


def main():
    print('=' * 70)
    print('STANDALONE ERQ (AQER) CORRECTION CORRECTNESS TEST')
    print('=' * 70)

    in_features, out_features = 384, 1536  # roughly fc1-sized
    n_calib, n_test = 4000, 2000
    print(f'\nSynthetic layer: {in_features} -> {out_features} (fc1-scale)')
    print(f'Calibration samples: {n_calib}, held-out test samples: {n_test}')
    print(f'(need N > in_features+1={in_features + 1} for a well-posed regression '
          f'— both splits satisfy this)')

    W_orig, b_orig = make_layer(in_features, out_features, seed=1)
    mixing, n_factors = make_mixing_matrix(in_features, seed=2)
    x_calib = make_correlated_inputs(n_calib, mixing, n_factors, seed=10)
    x_test = make_correlated_inputs(n_test, mixing, n_factors, seed=20)

    y_calib_true = x_calib @ W_orig.T + b_orig
    y_test_true = x_test @ W_orig.T + b_orig

    # ---- Check 1: zero quantization error -> near-zero correction ----
    print('\n[1/2] Sanity check: no quantization error -> correction should be ~0...')
    try:
        W_new_noop, b_new_noop = compute_aqer_correction(
            W_orig, b_orig, x_calib, y_calib_true, lam=0.1
        )
    except RuntimeError as e:
        print(f'  *** FAIL: {e}')
        return

    w_diff = (W_new_noop - W_orig).abs().max().item()
    b_diff = (b_new_noop - b_orig).abs().max().item()
    print(f'  Max |W_new - W_orig| = {w_diff:.8f} (should be ~0)')
    print(f'  Max |b_new - b_orig| = {b_diff:.8f} (should be ~0)')
    has_nan = torch.isnan(W_new_noop).any().item() or torch.isnan(b_new_noop).any().item()
    if has_nan:
        print('  *** FAIL: NaN in corrected weights. Stop.')
        return
    if w_diff > 0.01 or b_diff > 0.01:
        print('  *** WARNING: correction is not near-zero even with no quantization')
        print('  *** error to compensate for. Check the math/implementation before')
        print('  *** trusting the quantization-error test below.')
    else:
        print('  PASSED — correction is near-zero as expected when there is nothing')
        print('  to compensate for.')

    # ---- Check 2: real quantization error -> does correction generalize? ----
    print('\n[2/2] Generalization check: does Aqer reduce HELD-OUT MSE with real '
          'quantization error?')
    print(f'  {"Bits":>6} {"Lambda":>8} {"Baseline MSE":>14} {"Corrected MSE":>14} {"Improvement":>12}')
    all_improved = True
    for bits in [2, 4, 8]:
        x_calib_q, cal_min, cal_max = per_channel_quantize(x_calib, bits)
        x_test_q, _, _ = per_channel_quantize(x_test, bits, calib_min=cal_min, calib_max=cal_max)

        for lam in [0.01, 0.1, 1.0]:
            try:
                W_new, b_new = compute_aqer_correction(
                    W_orig, b_orig, x_calib_q, y_calib_true, lam=lam
                )
            except RuntimeError as e:
                print(f'  {bits:>6} {lam:>8} *** FAIL: {e}')
                continue

            baseline_pred = x_test_q @ W_orig.T + b_orig
            corrected_pred = x_test_q @ W_new.T + b_new

            baseline_mse = ((baseline_pred - y_test_true) ** 2).mean().item()
            corrected_mse = ((corrected_pred - y_test_true) ** 2).mean().item()

            improved = corrected_mse < baseline_mse
            all_improved = all_improved and improved
            improvement_pct = 100.0 * (1.0 - corrected_mse / baseline_mse) if baseline_mse > 0 else 0.0
            print(f'  {bits:>6} {lam:>8} {baseline_mse:>14.6f} {corrected_mse:>14.6f} '
                  f'{improvement_pct:>11.1f}%')

    print('\n' + '=' * 70)
    if all_improved:
        print('PASSED — Aqer correction reduces held-out MSE at every tested')
        print('bit-width and lambda. Worth proceeding to model wiring.')
    else:
        print('*** Some configurations did not improve on held-out data.')
        print('*** Investigate before proceeding — possible overfitting to the')
        print('*** calibration set, or lambda needs tuning per bit-width.')
    print('=' * 70)


if __name__ == '__main__':
    main()
