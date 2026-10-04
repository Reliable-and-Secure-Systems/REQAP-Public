"""
test_cgrp.py - unit tests for cgrp_pack_layer
Run: python test_cgrp.py
"""
import sys
from cgrp import cgrp_pack_layer, posthoc_pack_layer

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  PASS  {name}")
        PASSED += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED += 1


print("\n=== CGRP Unit Tests ===")

# Test 1: uniform channels - CGRP must match post-hoc exactly
print("\nTest 1: all same-width channels")
ch = [(4, 4)] * 8
r = cgrp_pack_layer(ch, R=16)
p = posthoc_pack_layer(ch, R=16)
check("n_regs equals post-hoc", r["n_regs"] == p["n_regs"],
      f"cgrp={r['n_regs']} posthoc={p['n_regs']}")
check("packed_issue_reduction equals post-hoc",
      abs(r["packed_issue_reduction"] - p["packed_issue_reduction"]) < 0.01)
check("zero_padding_waste is 0", r["zero_padding_waste"] == 0)

# Test 2: heterogeneous - CGRP must be <= post-hoc
print("\nTest 2: heterogeneous channels")
ch = [(4, 4), (8, 8), (8, 8), (4, 4)]
r = cgrp_pack_layer(ch, R=16)
p = posthoc_pack_layer(ch, R=16)
check("n_regs <= post-hoc", r["n_regs"] <= p["n_regs"],
      f"cgrp={r['n_regs']} posthoc={p['n_regs']}")
check("zero_padding_waste is 0", r["zero_padding_waste"] == 0)
check("fill_rate in (0, 1]", 0.0 < r["fill_rate"] <= 1.0)

# Test 3: single channel
print("\nTest 3: single channel")
r = cgrp_pack_layer([(4, 4)], R=16)
check("n_regs = 1", r["n_regs"] == 1)
check("fill_rate = 0.25", r["fill_rate"] == 0.25, str(r["fill_rate"]))
check("packed_issue_reduction = 1.0", r["packed_issue_reduction"] == 1.0)

# Test 4: overflow constraint - two 8-bit channels cannot share R=16
print("\nTest 4: overflow constraint")
# (2^8-1)^2 * 2 = 130050 > 2^16=65536 AND 8+8=16=R (storage exactly full)
# storage check: 8+8=16 <= 16 passes, BUT overflow: (255*255)*2 = 130050 >= 65536 FAILS
r = cgrp_pack_layer([(8, 8), (8, 8)], R=16)
check("two 8-bit channels cannot share R=16", r["n_regs"] == 2,
      f"n_regs={r['n_regs']} (expected 2, overflow constraint must fire)")

# Test 5: output structure
print("\nTest 5: output structure")
r = cgrp_pack_layer([(4, 4), (4, 4), (4, 4)], R=16)
check("bins is list", isinstance(r["bins"], list))
check("each bin is list", all(isinstance(b, list) for b in r["bins"]))
check("elements are 2-tuples",
      all(isinstance(c, tuple) and len(c) == 2
          for b in r["bins"] for c in b))

# Test 6: exact joint config example from brief
print("\nTest 6: joint config example - layer2.0.conv1")
cfg = {"weight": [4, 8, 8, 4], "activation": [4, 8, 8, 4]}
ch = list(zip(cfg["weight"], cfg["activation"]))
r = cgrp_pack_layer(ch, R=16)
p = posthoc_pack_layer(ch, R=16)
check("n_regs >= 1", r["n_regs"] >= 1)
check("CGRP n_regs <= post-hoc", r["n_regs"] <= p["n_regs"],
      f"cgrp={r['n_regs']} posthoc={p['n_regs']}")
check("zero_padding_waste = 0", r["zero_padding_waste"] == 0)
print(f"    CGRP:     n_regs={r['n_regs']} fill={r['fill_rate']} PIR={r['packed_issue_reduction']}")
print(f"    post-hoc: n_regs={p['n_regs']} fill={p['fill_rate']} PIR={p['packed_issue_reduction']}")

print(f"\n=== {PASSED} passed, {FAILED} failed ===")
if FAILED:
    print("STOP - fix failures before Task 4")
    sys.exit(1)
print("All tests passed - proceed to Task 4")
