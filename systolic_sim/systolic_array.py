"""
systolic_array.py

Weight-stationary systolic array simulator.

DATAFLOW (Weight-Stationary)
-----------------------------
Array: M rows x N columns of PEs.

Mapping GEMM  C[P x Q] = A[P x K] x B[K x Q]  onto the array:
  - P  = output neurons (C_out for conv, out_features for FC)
  - K  = input depth    (C_in*kH*kW for conv, in_features for FC)
  - Q  = spatial positions (H_out*W_out for conv, 1 for FC)

Outer loop: output-neuron tiles  t_p in range(ceil(P/M))
Inner loop: spatial tiles      t_q in range(ceil(Q/N))
    For each tile:
      Phase 1 - Load: fill M PE rows with their weight register words
                      (one RegisterWord per bin; weight stays stationary)
      Phase 2 - Stream: for each bin b in the packing result:
                        cycle += 1; every PE(row, col) multiplies its
                        weight field by the incoming activation field and
                        accumulates into its 32-bit register
      Phase 3 - Drain: harvest M*N accumulator values => output tile

CYCLE COUNTING
--------------
  Cycles per (p_tile, q_tile) pair = n_bins   (streaming steps)
                                   + (M - 1)   (wavefront startup)
  Total cycles per layer = ceil(P/M) * ceil(Q/N) * (n_bins + M - 1)

  In baseline: n_bins = K  (every slot is its own register word, d=1)
  In packed:   n_bins < K  (multiple slots per word,           d>=1)
  MACs are ALWAYS = P * Q * K  (invariant across both modes)

TRACE
-----
Up to MAX_TRACE_ROWS rows are recorded (to keep files manageable).
Each row covers one (cycle, pe_row, pe_col) triple.
"""

import numpy as np
from dataclasses import dataclass, field
from math import ceil
from typing import List, Optional, Dict, Any

from packing import PackingResult, RegisterWord

MAX_TRACE_ROWS = 200    # cap per-layer trace size (keeps files small; all layers get rows)


# ── PE state ─────────────────────────────────────────────────────────────────

@dataclass
class PEState:
    """Snapshot of one PE after one streaming step."""
    row:              int
    col:              int
    accumulator:      int   = 0      # 32-bit signed accumulator
    macs_done:        int   = 0
    overflow_risk:    bool  = False  # accumulator > 2^31 - 1 at any point


# ── Trace row ─────────────────────────────────────────────────────────────────

@dataclass
class TraceRow:
    """One row of the cycle-level execution trace."""
    cycle:          int
    layer:          str
    tile_id:        int
    pe_row:         int
    pe_col:         int
    bin_id:         int
    d:              int     # packing depth of this register word
    act_value:      int     # representative activation integer (field 0)
    weight_value:   int     # representative weight integer (field 0)
    product:        int     # product for field 0 only
    partial_sum:    int     # accumulator value after this step
    issue_id:       int     # sequential issue number within this layer
    lane_details:   str     # details of parallel lanes, e.g. "[0:7]: 5*2=10 + [8:11]: 1*3=3"

    def to_dict(self) -> dict:
        return {
            "cycle":        self.cycle,
            "layer":        self.layer,
            "tile_id":      self.tile_id,
            "pe_row":       self.pe_row,
            "pe_col":       self.pe_col,
            "bin_id":       self.bin_id,
            "d":            self.d,
            "act_value":    self.act_value,
            "weight_value": self.weight_value,
            "lane_products": self.lane_details,
            "product_sum":  self.product,
            "partial_sum":  self.partial_sum,
            "packed_issue_id": self.issue_id,
        }


# ── Per-tile result ───────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    """Spatial state of the entire array at a specific cycle (first tile only)."""
    cycle:          int
    bin_idx:        int      # current bin index entering column 0
    A:              np.ndarray  # (M, N) current activation values in PEs
    C:              np.ndarray  # (M, N) current accumulator values in PEs
    W:              np.ndarray  # (M, N) stationary weight values in PEs

@dataclass
class TileResult:
    tile_id:      int
    p_start:      int
    q_start:      int
    issues:       int   # packed register word reads (= n_bins)
    macs:         int   # issues * sum(d per bin)  == K*M_eff*N_eff
    cycles:       int   # issues + (M-1) wavefront


# ── Layer result ──────────────────────────────────────────────────────────────

@dataclass
class LayerSimResult:
    """Full simulation accounting for one layer."""
    layer_name:           str
    mode:                 str
    P:                    int    # output neurons
    Q:                    int    # spatial positions
    K:                    int    # input depth
    M:                    int    # array rows
    N:                    int    # array cols
    n_tiles_p:            int
    n_tiles_q:            int
    n_bins:               int    # packed words per K sweep
    total_issues:         int    # packed instruction count
    total_macs:           int    # total multiply-accumulate operations
    total_cycles:         int
    weight_words_read:    int    # total weight register reads
    act_words_read:       int    # total activation register reads
    avg_d:                float
    avg_fill_rate:        float
    wasted_bits:          int
    overflow_warnings:    int
    acc_max_magnitude:    int    # peak accumulator value observed
    parity_ok:            bool   = True
    parity_max_diff:      int    = 0
    snapshots:            List[Snapshot] = field(default_factory=list)
    packing:              Optional[PackingResult] = None
    trace:                List[TraceRow] = field(default_factory=list)
    tiles:                List[TileResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "layer":              self.layer_name,
            "mode":               self.mode,
            "K":                  self.K,
            "n_bins":             self.n_bins,
            "total_issues":       self.total_issues,
            "total_macs":         self.total_macs,
            "total_cycles":       self.total_cycles,
            "weight_words_read":  self.weight_words_read,
            "act_words_read":     self.act_words_read,
            "avg_fill_rate":      round(self.avg_fill_rate, 4),
            "wasted_bits":        self.wasted_bits,
            "overflow_warnings":  self.overflow_warnings,
            "acc_max_magnitude":  self.acc_max_magnitude,
        }


# ── Main simulation function ──────────────────────────────────────────────────

def simulate_layer(
    layer_name:   str,
    mode:         str,
    packing:      PackingResult,
    weight_mat:   np.ndarray,    # (P, K) quantized int32 weights
    act_mat:      np.ndarray,    # (K, Q) quantized int32 activations
    M:            int = 4,
    N:            int = 4,
    acc_bits:     int = 32,
) -> LayerSimResult:
    """
    Simulate systolic array execution for one GEMM layer.

    weight_mat: (P, K)  — rows are output neurons, cols are K-depth slots
    act_mat:    (K, Q)  — rows are K-depth slots, cols are spatial positions
    """
    P, K = weight_mat.shape
    K2, Q = act_mat.shape
    assert K == K2, f"K mismatch: weight K={K} vs act K={K2}"
    assert packing.n_slots == K, f"Packing slots {packing.n_slots} != K={K}"

    n_tiles_p = ceil(P / M)
    n_tiles_q = ceil(Q / N)
    n_bins    = packing.n_words
    acc_max   = 2 ** (acc_bits - 1) - 1

    result = LayerSimResult(
        layer_name=layer_name, mode=mode,
        P=P, Q=Q, K=K, M=M, N=N,
        n_tiles_p=n_tiles_p, n_tiles_q=n_tiles_q,
        n_bins=n_bins,
        total_issues=0, total_macs=0, total_cycles=0,
        weight_words_read=0, act_words_read=0,
        avg_d=packing.avg_d,
        avg_fill_rate=packing.avg_fill_rate,
        wasted_bits=packing.total_wasted_bits,
        overflow_warnings=packing.overflow_warnings,
        acc_max_magnitude=0,
        packing=packing,
    )

    global_cycle  = 0
    global_issue  = 0
    tile_id       = 0
    trace_rows    = []
    snapshots     = []

    for tp in range(n_tiles_p):
        p0 = tp * M
        p1 = min(p0 + M, P)
        M_eff = p1 - p0             # actual rows used in this tile

        for tq in range(n_tiles_q):
            q0 = tq * N
            q1 = min(q0 + N, Q)
            N_eff = q1 - q0         # actual cols used in this tile

            # ── Phase 1: Weight load ──────────────────────────────────────
            # Each PE row (output neuron p0..p1-1) pre-loads its weight words.
            # Cost: n_bins weight register reads per PE row.
            result.weight_words_read += M_eff * n_bins

            # ── Phase 2: Streaming ────────────────────────────────────────
            # For each register bin, stream one packed word per active column.
            tile_issues = 0
            tile_macs   = 0
            # PE accumulators for this tile: (M_eff, N_eff) -> int
            pe_acc = np.zeros((M_eff, N_eff), dtype=np.int64)

            for bin_idx, word in enumerate(packing.words):
                d = word.d
                tile_issues       += 1
                global_issue      += 1
                global_cycle      += 1
                result.act_words_read += N_eff   # one activation word per column

                # ── Vectorized MAC Logic ──────────────────────────────────
                # pe_acc is (M_eff, N_eff)
                # For each field for the current bin (register word):
                #   Accumulator += Weight_Col(k) @ Activation_Row(k)
                for field_obj in word.fields:
                    k_idx = field_obj.slot_index
                    # (M_eff, 1) @ (1, N_eff) -> (M_eff, N_eff)
                    w_vec = weight_mat[p0:p1, k_idx].reshape(-1, 1).astype(np.int64)
                    a_vec = act_mat[k_idx, q0:q1].reshape(1, -1).astype(np.int64)
                    pe_acc += w_vec @ a_vec
                
                tile_macs += d * (M_eff * N_eff)

                # ── Metrics & Trace Sampling ──────────────────────────────
                current_max = np.max(np.abs(pe_acc))
                result.acc_max_magnitude = max(result.acc_max_magnitude, int(current_max))
                
                # Check for overflows in this bin
                if current_max > acc_max:
                    # Count how many PEs overflowed in this step
                    result.overflow_warnings += np.sum(np.abs(pe_acc) > acc_max)

                # Record trace (sample only the first PE of the tile for efficiency)
                if len(trace_rows) < MAX_TRACE_ROWS and tile_id % 10 == 0:
                    pr, pc = 0, 0
                    p_neuron, q_pos = p0 + pr, q0 + pc
                    f0 = word.fields[0]
                    # Compute lane string for the trace display
                    lane_parts = []
                    for f in word.fields:
                        w_v = int(weight_mat[p_neuron, f.slot_index])
                        a_v = int(act_mat[f.slot_index, q_pos])
                        lane_parts.append(f"[{f.bit_offset}:{f.bit_offset + f.weight_bits - 1}]: {w_v}*{a_v}={w_v*a_v}")

                    trace_rows.append(TraceRow(
                        cycle=global_cycle, layer=layer_name, tile_id=tile_id,
                        pe_row=pr, pe_col=pc, bin_id=bin_idx, d=d,
                        act_value=int(act_mat[f0.slot_index, q_pos]),
                        weight_value=int(weight_mat[p_neuron, f0.slot_index]),
                        product=int(pe_acc[pr, pc]), # This is the full local acc
                        partial_sum=int(pe_acc[pr, pc]),
                        issue_id=global_issue,
                        lane_details=" + ".join(lane_parts),
                    ))


                # ── Record Spatial Snapshot (First Tile Only) ────────────────
                if tile_id == 0:
                    # In WS systolic arrays, data at PE(r,c) is delayed by c cycles
                    # So at local_cycle t, Column c sees Bin (t - c)
                    snap_A = np.zeros((M_eff, N_eff), dtype=np.int32)
                    snap_W = np.zeros((M_eff, N_eff), dtype=np.int32)
                    
                    t = bin_idx
                    for pc in range(N_eff):
                        k_word_idx = t - pc
                        if 0 <= k_word_idx < n_bins:
                            word_at_col = packing.words[k_word_idx]
                            # Use first field's activation as representative for display
                            f0 = word_at_col.fields[0]
                            for pr in range(M_eff):
                                snap_A[pr, pc] = int(act_mat[f0.slot_index, q0 + pc])
                                snap_W[pr, pc] = int(weight_mat[p0 + pr, f0.slot_index])
                    
                    snapshots.append(Snapshot(
                        cycle=global_cycle,
                        bin_idx=bin_idx,
                        A=snap_A.copy(),
                        C=pe_acc.copy(),
                        W=snap_W.copy()
                    ))

            # Wavefront: M_eff - 1 extra cycles for systolic propagation
            tile_cycles = tile_issues + (M_eff - 1)

            result.total_issues += tile_issues
            result.total_macs   += tile_macs
            result.total_cycles += tile_cycles
            result.tiles.append(TileResult(
                tile_id=tile_id,
                p_start=p0, q_start=q0,
                issues=tile_issues,
                macs=tile_macs,
                cycles=tile_cycles,
            ))
            tile_id += 1

    result.trace = trace_rows
    result.snapshots = snapshots
    return result
