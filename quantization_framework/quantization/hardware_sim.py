import torch
import math
try:
    from quantization.packing import ReQAPPackingPlanner
except ModuleNotFoundError:
    from .packing import ReQAPPackingPlanner

class RegisterPackingSimulator:
    """
    SIMD packing hardware simulator.
    Evaluates Safe-FFD safety bounds against physical register widths.
    """
    def __init__(self, register_size=16):
        self.R = register_size
        self._planner = ReQAPPackingPlanner(register_size=register_size, max_d=8)

    def is_valid_packing(self, w_bits, a_bits, d):
        """Evaluate Eq.1 safety bound for (W, A, d)."""
        return self._planner.is_safe(w_bits, a_bits, d)

    def find_max_packing_factor(self, w_bits, a_bits, max_d=8):
        """Calculate maximum Safe-FFD packing density (d)."""
        self._planner.max_d = int(max_d)
        return self._planner.best_factor(w_bits, a_bits)

    def get_carrying_budget(self, w_bits, a_bits, d):
        """Calculate residual accumulator margin post-MAC (in bits)."""
        if d <= 0: return 0
        segment_size = self.R // d
        
        # Max value of a single multiplication
        max_prod = (2**w_bits - 1) * (2**a_bits - 1)
        if max_prod == 0: return segment_size
        
        # Bits needed for product
        prod_bits = math.ceil(math.log2(max_prod + 1))
        
        # Residual bits for carrying (accumulation)
        carrying_bits = segment_size - prod_bits
        return max(0, carrying_bits)

    def get_packing_efficiency(self, w_bits, a_bits):
        """Calculate lane utilization metrics."""
        d = self.find_max_packing_factor(w_bits, a_bits)
        segment_size = self.R // d
        
        bits_used = w_bits + a_bits
        utilization = (bits_used / segment_size) * 100
        carry_budget = self.get_carrying_budget(w_bits, a_bits, d)
        
        return d, utilization, carry_budget

    def calculate_register_savings(self, layer_config, num_params):
        """Compute static memory footprint delta."""
        w_bits = layer_config.get('weight', 8)
        
        total_bits_fp32 = num_params * 32
        total_bits_int8 = num_params * 8
        total_bits_hrp = num_params * w_bits
        
        return (total_bits_fp32 - total_bits_hrp), (total_bits_int8 - total_bits_hrp)

def calculate_model_throughput(model_config, register_size=16):
    """
    Calculate total throughput gain across all layers.
    Throughput = Sum(MACs_i * d_i) / Sum(MACs_i)
    """
    sim = RegisterPackingSimulator(register_size)
    total_weighted_d = 0
    total_macs = 0
    
    # This requires layer MAC info which we'll get from the engine
    # For now, this is a skeleton for the search engine to use
    pass
