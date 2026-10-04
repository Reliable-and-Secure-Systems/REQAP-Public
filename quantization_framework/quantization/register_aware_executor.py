import torch
try:
    from quantization.packing import ReQAPPackingPlanner
except ModuleNotFoundError:
    from .packing import ReQAPPackingPlanner

# HW SIM: Global execution state
MQF_GLOBAL_CONTEXT = {
    'last_scale': 1.0,
    'is_first_layer': True,
    'register_size': 16,
    'carrying_bits': 0  # Number of bits reserved for carry/accumulation
}

def get_carrying_budget(packing_factor, w_bits, a_bits, register_size=16):
    """Query remaining accumulator margin post-MAC via planner."""
    planner = ReQAPPackingPlanner(register_size=register_size, max_d=max(1, packing_factor))
    plan = planner.plan(w_bits, a_bits, d=packing_factor)
    return plan.lane_headroom_bits
