# Systolic Array Verification Report
Generated on: 2026-04-06 01:03:37

## 1. Hand-Verifiable Example Definition

K=4 input slots, P=2 neurons, Q=1 spatial position

**Weight Matrix W:**
```
    neuron 0: [3, -2, 5, 1]
    neuron 1: [-1, 4, -3, 2]
```

**Activation Column A:**
```
    slot 0: 7
    slot 1: 2
    slot 2: -4
    slot 3: 3
```

**Reference Calculation:**
```
    neuron 0: (3*7) + (-2*2) + (5*-4) + (1*3) = 0
    neuron 1: (-1*7) + (4*2) + (-3*-4) + (2*3) = 19
```

## 2. Mode Verification (Baseline vs Packed)

| Mode | Cycles | d | Match? |
|---|---|---|---|
| Baseline | 4 | 1 | PASS |
| Packed | 1 | 2 | PASS |

**Conclusion:** Packed mode achieves correct result in 1 cycles vs 4 cycles.

## 3. Detailed PE Register Flow (Neuron 0)

This table shows how bits are unpacked and summed in one cycle.

| Word | Field | Kslot | w_bits | a_bits | w_val | a_val | product | running partial | acc after word |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 4 | 4 | 3 | 7 | 21 | 21 | |
| 1 | 1 | 1 | 4 | 4 | -2 | 2 | -4 | 17 | |
| 1 | 2 | 2 | 4 | 4 | 5 | -4 | -20 | -3 | |
| 1 | 3 | 3 | 4 | 4 | 1 | 3 | 3 | 0 | |
| 1 | ->ACC | | | | | | | | **0** |


## 4. Overflow Safety Proof

Constraint: `sum( (2^bw - 1) * (2^ba - 1) ) < 2^R`

| Configuration | LHS (Worst Case) | 2^R=65536 | Safe? | d |
|---|---|---|---|---|
| 8b alone (baseline) | 65,025 | 65,536 | YES | 1 |
| Two 8b together | 130,050 | 65,536 | NO | 2 |
| 4b + 4b packed | 450 | 65,536 | YES | 2 |
| 4b x 4 (four per word) | 900 | 65,536 | YES | 4 |
| 4b + 8b mixed | 65,250 | 65,536 | YES | 2 |


## 5. Real Layer Validation (CNN conv1)

Comparing simulator integer accumulator vs NumPy int64 matrix multiplication.

| Mode | Max Acc Error vs NumPy |
|---|---|
| Baseline (8b) | 0 |
| Packed (Mixed) | 0 |


## 6. MAC Invariant Proof

Verifying that total MAC work remains constant.

| Layer | Expected MACs | Base MACs | OK? | Pack MACs | OK? |
|---|---|---|---|---|---|
| conv1 | 56,448 | 56,448 | True | 56,448 | True |
| conv2 | 225,792 | 225,792 | True | 225,792 | True |
| fc1 | 7,840 | 7,840 | True | 7,840 | True |


## 7. Summary

1. **Mathematical Equivalence**: Both baseline and packed compute same dot product.
2. **Safety**: Register packing obeys overflow constraints.
3. **Invariance**: Total MAC work is identical; only register reads/cycles are reduced.

**STATUS: ALL VERIFICATIONS PASS**
