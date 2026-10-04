"""
audit.py

Utility to trace the numerical lineage of the first output element of a layer.
Provides a step-by-step breakdown of:
  [quantized_input] * [quantized_weight] = [partial_sum]
Used for human-level verification in the thesis PoC.
"""
import numpy as np
from typing import Dict, List, Any

def audit_layer_first_element(
    layer_cfg:   dict,
    weight_mat:  np.ndarray,    # (P, K) quantized
    act_mat:     np.ndarray,    # (K, Q) quantized
    w_scale:     float,
    a_scale:     float,
) -> Dict[str, Any]:
    """
    Extract the MAC-by-MAC lineage for Output[0, 0] (GEMM space).
    In GEMM, p=0 is the first output neuron, q=0 is the first spatial position.
    """
    P, K = weight_mat.shape
    _, Q = act_mat.shape
    
    # Target: Neuron 0, Position 0
    p = 0
    q = 0
    
    macs = []
    running_acc = 0
    
    # Unroll the K-length dot product
    for k in range(K):
        w_val = int(weight_mat[p, k])
        a_val = int(act_mat[k, q])
        prod  = w_val * a_val
        running_acc += prod
        
        # Only record first 20 MACs + every 100th if K is large (to save memory)
        if k < 20 or k % 100 == 0 or k == K-1:
            macs.append({
                "k": k,
                "w": w_val,
                "a": a_val,
                "prod": prod,
                "acc": running_acc
            })
            
    float_val = float(running_acc) * w_scale * a_scale
    
    return {
        "layer_name": layer_cfg["name"],
        "type":       layer_cfg["type"],
        "K":          K,
        "macs":       macs,
        "int_acc":    int(running_acc),
        "w_scale":    w_scale,
        "a_scale":    a_scale,
        "float_val":  float_val
    }
