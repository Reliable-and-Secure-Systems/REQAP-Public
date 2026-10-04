"""
metrics.py

Metrics computation, comparison, and output generation.

Exports
-------
compute_comparison(baseline_result, packed_result)  -> comparison dict
build_trace_df(layer_results)                       -> pd.DataFrame
build_packing_df(packing_results)                   -> pd.DataFrame
save_results_json(comparison, path)
save_trace_csv(trace_df, path)
"""

import json
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from systolic_array import LayerSimResult
from packing import PackingResult


# ── Layer-level metrics ───────────────────────────────────────────────────────

def _layer_metrics(res: LayerSimResult, mae: float) -> dict:
    return {
        "layer":              res.layer_name,
        "mode":               res.mode,
        "K":                  res.K,
        "P":                  res.P,
        "Q":                  res.Q,
        "n_bins":             res.n_bins,
        "total_issues":       res.total_issues,
        "total_macs":         res.total_macs,
        "total_cycles":       res.total_cycles,
        "weight_words_read":  res.weight_words_read,
        "act_words_read":     res.act_words_read,
        "avg_d":              res.avg_d,
        "avg_fill_rate_pct":  round(res.avg_fill_rate * 100, 2),
        "wasted_bits":        res.wasted_bits,
        "overflow_warnings":  res.overflow_warnings,
        "acc_max_magnitude":  res.acc_max_magnitude,
        "max_abs_error":      round(mae, 6),
        "parity_ok":          res.parity_ok,
        "parity_max_diff":    res.parity_max_diff,
    }


# ── Comparison ────────────────────────────────────────────────────────────────

def compute_comparison(baseline: dict, packed: dict) -> dict:
    """
    Build a structured comparison dict from two run_simulation() results.
    Includes per-layer metrics for both modes and overall totals.
    """
    layers = list(baseline["layer_results"].keys())

    baseline_layers = {}
    packed_layers   = {}
    delta_layers    = {}

    for lname in layers:
        # We only care about parity from the packed run (which does the comparison)
        prity = packed["parity_results"].get(lname, {"parity_ok": True, "max_diff": 0})
        
        bm = _layer_metrics(
            baseline["layer_results"][lname],
            baseline["max_abs_errors"].get(lname, 0.0),
        )
        # Attach parity specifically to the layer results
        bm["parity_ok"] = True # Baseline is always OK vs itself
        bm["max_diff"] = 0
        
        pm = _layer_metrics(
            packed["layer_results"][lname],
            packed["max_abs_errors"].get(lname, 0.0),
        )
        pm["parity_ok"] = prity["parity_ok"]
        pm["max_diff"] = prity["max_diff"]
        baseline_layers[lname] = bm
        packed_layers[lname]   = pm

        # Delta (packed - baseline); positive = packed is worse
        delta_layers[lname] = {
            k: (pm[k] - bm[k]) if isinstance(bm[k], (int, float)) else None
            for k in bm
        }

    tb = baseline["totals"]
    tp = packed["totals"]

    def pct(a, b):
        return round((b - a) / a * 100, 2) if a != 0 else 0.0

    totals_comparison = {
        "baseline":          tb,
        "packed":            tp,
        "delta": {
            k: {"abs": tp[k] - tb[k], "pct": pct(tb[k], tp[k])}
            for k in tb
        },
    }

    # MAC invariant and Parity check
    mac_invariant_ok = (tb["total_macs"] == tp["total_macs"])
    parity_ok        = all(pm["parity_ok"] for pm in packed_layers.values())

    return {
        "layers":            layers,
        "baseline_layers":   baseline_layers,
        "packed_layers":     packed_layers,
        "delta_layers":      delta_layers,
        "totals":            totals_comparison,
        "mac_invariant_ok":  mac_invariant_ok,
        "parity_ok":         parity_ok,
        "baseline_mae":      baseline["max_abs_errors"],
        "packed_mae":        packed["max_abs_errors"],
    }


# ── Trace DataFrame ───────────────────────────────────────────────────────────

def build_trace_df(layer_results: Dict[str, LayerSimResult],
                   max_rows: int = 2000) -> pd.DataFrame:
    """
    Build a combined DataFrame from all layer traces.

    Budget is allocated EQUALLY per layer so that no single layer
    (e.g. conv1) monopolises the row limit and leaves fc1 empty.
    Each layer contributes at most  max_rows // n_layers  rows.
    """
    rows: list = []
    n_layers = max(len(layer_results), 1)
    per_layer = max(max_rows // n_layers, 50)   # at least 50 per layer

    for res in layer_results.values():
        count = 0
        for tr in res.trace:
            rows.append(tr.to_dict())
            count += 1
            if count >= per_layer:
                break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_layer_trace_df(layer_results: Dict[str, LayerSimResult],
                         layer_name: str,
                         max_rows: int = 500) -> pd.DataFrame:
    """
    Build a trace DataFrame for ONE specific layer.
    Bypasses the global cap so the selected layer always shows its rows.
    Used by the UI when the layer dropdown is set to a specific layer.
    """
    res = layer_results.get(layer_name)
    if res is None:
        return pd.DataFrame()
    rows = [tr.to_dict() for tr in res.trace[:max_rows]]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Packing DataFrame ─────────────────────────────────────────────────────────

def build_packing_df(packing_results: Dict[str, PackingResult]) -> pd.DataFrame:
    """Build a flat DataFrame of RegisterWord summaries, one row per word."""
    rows = []
    for pr in packing_results.values():
        for word in pr.words:
            rows.append({
                "layer":      pr.layer_name,
                "mode":       pr.mode,
                "word_id":    word.word_id,
                "d":          word.d,
                "used_bits":  word.used_weight_bits,
                "slack_bits": word.slack_bits,
                "fill_rate":  round(word.fill_rate, 4),
                "overflow_ok": word.overflow_safe(),
                "label":      word.label(),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_packing_summary_df(comparison: dict) -> pd.DataFrame:
    """One-row-per-layer packing summary for both modes."""
    rows = []
    for lname, bm in comparison["baseline_layers"].items():
        pm = comparison["packed_layers"][lname]
        rows.append({
            "layer":                lname,
            "K_slots":              bm["K"],
            "baseline_bins":        bm["n_bins"],
            "packed_bins":          pm["n_bins"],
            "baseline_fill_pct":    bm["avg_fill_rate_pct"],
            "packed_fill_pct":      pm["avg_fill_rate_pct"],
            "baseline_wasted_bits": bm["wasted_bits"],
            "packed_wasted_bits":   pm["wasted_bits"],
            "baseline_ov_warn":     bm["overflow_warnings"],
            "packed_ov_warn":       pm["overflow_warnings"],
        })
    return pd.DataFrame(rows)


# ── File I/O ──────────────────────────────────────────────────────────────────

def save_results_json(comparison: dict, path: str) -> None:
    """Save the comparison dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Make JSON-serialisable (numpy ints -> python ints)
    def _fix(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_fix(comparison), f, indent=2)


def save_trace_csv(trace_df: pd.DataFrame, path: str) -> None:
    """Save the trace DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trace_df.to_csv(path, index=False)


# ── Convenience: run-and-export ───────────────────────────────────────────────

def generate_sample_outputs(results: dict, out_dir: str = "output") -> None:
    """Generate sample_results.json and sample_trace.csv from run_both_modes()."""
    comparison = compute_comparison(results["baseline"], results["packed"])
    save_results_json(comparison, os.path.join(out_dir, "sample_results.json"))

    trace_df = build_trace_df(results["baseline"]["layer_results"])
    save_trace_csv(trace_df, os.path.join(out_dir, "sample_trace.csv"))
    print(f"Saved outputs to {out_dir}/")
