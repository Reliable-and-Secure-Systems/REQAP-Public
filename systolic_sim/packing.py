"""
packing.py

Register packing algorithms for systolic array weight/activation packing.

OVERVIEW
--------
A "bin" (RegisterWord) groups one or more K-dimension operand slots into a
single R-bit register word.  Each slot carries (weight_bits, act_bits).

Two packing strategies
  1. Naive post-hoc  : d = floor(R / max_weight_bits). Fast; may violate overflow.
  2. Safe FFD        : First-Fit Decreasing with storage-fit + overflow checks.

Baseline (Path A) always assigns d=1 (each slot alone) because two 8-bit
channels fail the overflow check: (255*255)*2 = 130050 > 65536.
"""
from dataclasses import dataclass, field
from math import floor
from typing import List, Tuple


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class OpdField:
    """One operand packed inside a register word."""
    bit_offset:  int    # start bit within the R-bit register word
    weight_bits: int    # bits allocated for the weight value
    act_bits:    int    # bits allocated for the activation value
    slot_index:  int    # original index in the K-length channel list


@dataclass
class RegisterWord:
    """A packed R-bit register word holding one or more operand fields."""
    word_id: int
    r:       int                               # register width in bits
    fields:  List[OpdField] = field(default_factory=list)

    @property
    def d(self) -> int:
        """Packing depth: how many operand pairs are packed into this word."""
        return len(self.fields)

    @property
    def used_weight_bits(self) -> int:
        return sum(f.weight_bits for f in self.fields)

    @property
    def slack_bits(self) -> int:
        return self.r - self.used_weight_bits

    @property
    def fill_rate(self) -> float:
        return self.used_weight_bits / self.r if self.r > 0 else 0.0

    def overflow_safe(self) -> bool:
        """
        Conservative heterogeneous overflow check.
        Guarantees the 32-bit accumulator cannot roll over when summing
        partial products from all fields in this word.
            sum_i( (2^bw_i - 1) * (2^ba_i - 1) ) < 2^R
        """
        lhs = sum((2 ** f.weight_bits - 1) * (2 ** f.act_bits - 1)
                  for f in self.fields)
        return lhs < 2 ** self.r

    def label(self) -> str:
        parts = [f"{f.weight_bits}w/{f.act_bits}a" for f in self.fields]
        return "[" + ", ".join(parts) + "]"

    def to_dict(self) -> dict:
        return {
            "word_id":      self.word_id,
            "d":            self.d,
            "used_bits":    self.used_weight_bits,
            "slack_bits":   self.slack_bits,
            "fill_rate":    round(self.fill_rate, 4),
            "overflow_ok":  self.overflow_safe(),
            "label":        self.label(),
            "fields": [
                {"offset": f.bit_offset, "w_bits": f.weight_bits,
                 "a_bits": f.act_bits,   "slot": f.slot_index}
                for f in self.fields
            ],
        }


@dataclass
class PackingResult:
    """Packing result for one layer."""
    layer_name: str
    mode:       str           # "baseline" | "naive" | "safe"
    r:          int
    n_slots:    int           # total K slots
    words:      List[RegisterWord] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def reduction_ratio(self) -> float:
        """n_words / n_slots — lower is better (more packing)."""
        return self.n_words / self.n_slots if self.n_slots > 0 else 1.0

    @property
    def total_wasted_bits(self) -> int:
        return sum(w.slack_bits for w in self.words)

    @property
    def avg_fill_rate(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.fill_rate for w in self.words) / len(self.words)

    @property
    def avg_d(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.d for w in self.words) / len(self.words)

    @property
    def overflow_warnings(self) -> int:
        return sum(1 for w in self.words if not w.overflow_safe())

    def summary(self) -> dict:
        return {
            "layer":            self.layer_name,
            "mode":             self.mode,
            "n_slots":          self.n_slots,
            "n_words":          self.n_words,
            "reduction_ratio":  round(self.reduction_ratio, 4),
            "wasted_bits":      self.total_wasted_bits,
            "avg_fill_rate":    round(self.avg_fill_rate, 4),
            "avg_d":            round(self.avg_d, 3),
            "overflow_warnings": self.overflow_warnings,
        }


# ── Constraint predicates ─────────────────────────────────────────────────────

def _storage_fits(slots: List[Tuple[int, int]], r: int) -> bool:
    """Total weight bits must not exceed register width."""
    return sum(bw for bw, _ in slots) <= r


def _overflow_safe(slots: List[Tuple[int, int]], r: int) -> bool:
    """Conservative overflow check for a candidate bin."""
    lhs = sum((2 ** bw - 1) * (2 ** ba - 1) for bw, ba in slots)
    return lhs < 2 ** r


# ── Packing algorithms ────────────────────────────────────────────────────────

def pack_baseline(channels: List[Tuple[int, int]], r: int,
                  layer_name: str) -> PackingResult:
    """
    Baseline: every slot is alone (d = 1).
    Correct for 8-bit uniform mode where two channels always fail overflow.
    """
    result = PackingResult(layer_name=layer_name, mode="baseline",
                           r=r, n_slots=len(channels))
    for idx, (bw, ba) in enumerate(channels):
        word = RegisterWord(word_id=idx, r=r)
        word.fields.append(
            OpdField(bit_offset=0, weight_bits=bw, act_bits=ba, slot_index=idx))
        result.words.append(word)
    return result


def pack_naive(channels: List[Tuple[int, int]], r: int,
               layer_name: str) -> PackingResult:
    """
    Naive post-hoc packing: d = floor(R / max_weight_bits).
    Groups d consecutive channels into one word.
    Overflow safety is NOT guaranteed; warnings are flagged in the result.
    """
    result = PackingResult(layer_name=layer_name, mode="naive",
                           r=r, n_slots=len(channels))
    if not channels:
        return result
    max_bw = max(bw for bw, _ in channels)
    d = max(1, floor(r / max_bw))
    word_id = 0
    for i in range(0, len(channels), d):
        group = channels[i:i + d]
        word = RegisterWord(word_id=word_id, r=r)
        offset = 0
        for j, (bw, ba) in enumerate(group):
            word.fields.append(
                OpdField(bit_offset=offset, weight_bits=bw, act_bits=ba,
                         slot_index=i + j))
            offset += bw
        result.words.append(word)
        word_id += 1
    return result


def pack_safe_ffd(channels: List[Tuple[int, int]], r: int,
                  layer_name: str) -> PackingResult:
    """
    Safe First-Fit Decreasing bin packing.

    Algorithm
    ---------
    1. Sort slots by (weight_bits + act_bits) descending.
    2. For each slot, try to add it to the first existing bin where both
       storage_fit AND overflow_safe are satisfied.
    3. If no bin fits, open a new bin.

    Both constraints are enforced, so the result is always correct.
    """
    result = PackingResult(layer_name=layer_name, mode="safe",
                           r=r, n_slots=len(channels))
    if not channels:
        return result

    # Sort by total bit width descending (FFD order)
    sorted_items = sorted(enumerate(channels),
                          key=lambda x: x[1][0] + x[1][1], reverse=True)

    # Each bin: list of (orig_idx, bw, ba)
    bins: List[List[Tuple[int, int, int]]] = []

    for orig_idx, (bw, ba) in sorted_items:
        placed = False
        for b in bins:
            existing = [(x[1], x[2]) for x in b]
            candidate = existing + [(bw, ba)]
            if _storage_fits(candidate, r) and _overflow_safe(candidate, r):
                b.append((orig_idx, bw, ba))
                placed = True
                break
        if not placed:
            bins.append([(orig_idx, bw, ba)])

    for word_id, b in enumerate(bins):
        word = RegisterWord(word_id=word_id, r=r)
        offset = 0
        for orig_idx, bw, ba in b:
            word.fields.append(
                OpdField(bit_offset=offset, weight_bits=bw, act_bits=ba,
                         slot_index=orig_idx))
            offset += bw
        result.words.append(word)

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def pack_layer(layer_name: str, quant_cfg: dict, r: int,
               mode: str) -> PackingResult:
    """
    Pack one layer's K slots according to mode.

    mode:
      "baseline" -> pack_baseline  (d = 1 always)
      "naive"    -> pack_naive     (d = floor(R/max_bw), no overflow check)
      "safe"     -> pack_safe_ffd  (FFD with storage + overflow constraints)
    """
    w_bits = quant_cfg["weight_bits"]
    a_bits = quant_cfg["act_bits"]
    channels = list(zip(w_bits, a_bits))

    if mode == "baseline":
        return pack_baseline(channels, r, layer_name)
    elif mode == "naive":
        return pack_naive(channels, r, layer_name)
    else:
        return pack_safe_ffd(channels, r, layer_name)
