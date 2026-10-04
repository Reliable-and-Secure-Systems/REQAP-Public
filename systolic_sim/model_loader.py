import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# ── Path Setup ────────────────────────────────────────────────────────────────
# Ensure we can import from quantization_framework/models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMEWORK_ROOT = os.path.join(PROJECT_ROOT, "quantization_framework")

for path in [PROJECT_ROOT, FRAMEWORK_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from quantization_framework.models.alexnet import AlexNet
    from quantization_framework.models.resnet import ResNet18
    from quantization_framework.models.vgg import vgg11_bn
except ImportError as e:
    # Fallback for different environments or local debugging
    print(f"Warning: Could not import models from quantization_framework. Error: {e}")

# ── Configuration Files Mapping ───────────────────────────────────────────────
MODEL_FILES = {
    "Simple PoC": {
        "class": "SimplePoC",
        "json": None, # Use hardcoded [4,4,8]
        "input_shape": (1, 28, 28),
    },
    "AlexNet": {
        "class": "AlexNet",
        "json": "alexnet_config_2_4_8.json",
        "input_shape": (1, 227, 227), # AlexNet requires standard resolution
    },
    "ResNet18": {
        "class": "ResNet18",
        "json": "resnet18_config_2_4_8.json",
        "input_shape": (3, 32, 32),
    },
    "VGG11": {
        "class": "vgg11_bn",
        "json": "vgg11_bn_config_2_4_8.json",
        "input_shape": (3, 32, 32),
    }
}

class SimpleModel(nn.Module):
    """Recreate the original 3-layer PoC model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)




class ModelLoader:
    def __init__(self, model_name: str, seed: int = 42):
        if model_name not in MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.info = MODEL_FILES[model_name]
        
        # Load Architecture
        self.model = self._instantiate_model()
        self._seed_weights(seed)
        self.model.eval() # Ensure deterministic forward pass

        # Load Quantization Config
        if self.info["json"]:
            self.json_path = self._resolve_json_path(self.info["json"])
            with open(self.json_path, "r") as f:
                data = json.load(f)
                self.quant_json = data.get("config", data)
        else:
            self.quant_json = {} # Use cyclic defaults
            
        # Extract layers and shapes
        self.layers = self._extract_layers()

    def _resolve_json_path(self, json_name: str) -> str:
        """Robustly search for quantization config files across possible directories."""
        if not json_name:
            return None
        search_paths = [
            os.path.join(FRAMEWORK_ROOT, "configs", json_name),
            os.path.join(PROJECT_ROOT, "quantization_framework", "configs", json_name),
            os.path.join(PROJECT_ROOT, json_name),
            os.path.join(os.path.dirname(__file__), json_name),
            json_name
        ]
        for p in search_paths:
            if os.path.exists(p):
                return os.path.abspath(p)
        return os.path.join(FRAMEWORK_ROOT, "configs", json_name)

    def _seed_weights(self, seed: int):
        """Force deterministic weights for simulation parity."""
        rng = np.random.default_rng(seed)
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Match the old PoC's scaling (0.1)
                w_np = rng.standard_normal(m.weight.shape).astype(np.float32) * 0.1
                m.weight.data = torch.from_numpy(w_np)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _instantiate_model(self):
        if self.model_name == "Simple PoC":
            return SimpleModel()
        elif self.model_name == "AlexNet":
            return AlexNet(num_classes=10)
        elif self.model_name == "ResNet18":
            return ResNet18(num_classes=43) 
        elif self.model_name == "VGG11":
            return vgg11_bn(num_classes=10)
        return None


    def _extract_layers(self) -> List[dict]:
        """
        Iterate through model modules to find Conv and Linear layers.
        Compute K, P, Q for each.
        """
        layers = []
        
        # We need spatial dimensions. We'll use a dummy pass or calculated logic.
        # Calculated logic is safer for PoC.
        input_shape = self.info["input_shape"]
        curr_c, curr_h, curr_w = input_shape
        
        for name, module in self.model.named_modules():
            # Match only weight-bearing layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):

                l_info = {
                    "name": name,
                    "type": "conv" if isinstance(module, nn.Conv2d) else "fc",
                    "module": module,
                }
                
                if isinstance(module, nn.Conv2d):
                    C_out = module.out_channels
                    C_in  = module.in_channels
                    kH, kW = module.kernel_size
                    stride = module.stride[0]
                    padding = module.padding[0]
                    
                    # Compute spatial dimensions (assuming standard formula)
                    H_out = (curr_h + 2*padding - kH) // stride + 1
                    W_out = (curr_w + 2*padding - kW) // stride + 1
                    
                    l_info.update({
                        "K": C_in * kH * kW,
                        "P": C_out,
                        "Q": H_out * W_out,
                        "in_ch": C_in,
                        "out_ch": C_out,
                        "kernel": kH,
                        "stride": stride,
                        "padding": padding,
                        "h_out": H_out,
                        "w_out": W_out
                    })
                    # Update curr_h, curr_w for next layers? 
                    # Only if it's the main backbone. For ResNet this is complex.
                    # Simplification: We assume the JSON/Model are consistent.
                    # Actually, we should track spatial dims layer by layer.
                    curr_h, curr_w = H_out, W_out
                    curr_c = C_out
                else: # Linear
                    P = module.out_features
                    K = module.in_features
                    l_info.update({
                        "K": K,
                        "P": P,
                        "Q": 1,
                        "in_features": K,
                        "out_features": P
                    })
                
                # Add bit-width info from JSON or defaults
                if name in self.quant_json:
                    q_cfg = self.quant_json[name]
                else:
                    # Default cyclic [4, 4, 8] pattern if no JSON provided
                    q_cfg = {"weight": [4, 4, 8], "activation": [4, 4, 8]}

                l_info["quant"] = self._expand_bitwidths(q_cfg, l_info["K"], l_info)
                
                layers.append(l_info)

            
            # Handle Pooling/ReLU to update spatial dimensions if they are NOT weight bearing
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    # For adaptive pool, output size is fixed
                    if isinstance(module.output_size, int):
                        curr_h, curr_w = module.output_size, module.output_size
                    else:
                        curr_h, curr_w = module.output_size
                else:
                    # Generic formula: (in + 2*p - k) // s + 1
                    kh = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                    st = module.stride if isinstance(module.stride, int) else module.stride[0]
                    pa = module.padding if isinstance(module.padding, int) else module.padding[0]
                    # Note: dilation is usually 1 for pooling
                    curr_h = (curr_h + 2*pa - kh) // st + 1
                    curr_w = (curr_w + 2*pa - kh) // st + 1


        return layers

    def _expand_bitwidths(self, q_cfg: dict, K: int, l_info: dict) -> dict:
        """
        Expand bit-width definitions to full K-length lists.
        Handles:
        1. Single integer (broadcast to K)
        2. List of length C_in (expand by kernel_size^2)
        3. List of length K (use as is)
        """
        def expand_v(val, K_len, layer_meta):
            if isinstance(val, (int, float)):
                return [int(val)] * K_len
            if isinstance(val, list):
                if len(val) == K_len:
                    return val
                # Check if it's per-channel
                in_ch = layer_meta.get("in_ch")
                if in_ch and len(val) == in_ch:
                    kernel_area = layer_meta.get("kernel", 1)**2
                    expanded = []
                    for v in val:
                        expanded.extend([v] * kernel_area)
                    # If there's a mismatch (padding/groups), clip or pad
                    return (expanded[:K_len] + [expanded[-1]] * (K_len - len(expanded)))[:K_len]
                # Default: cyclic repeat if it's some other granular size
                return [val[i % len(val)] for i in range(K_len)]
            return [8] * K_len

        return {
            "weight_bits": expand_v(q_cfg["weight"], K, l_info),
            "act_bits":    expand_v(q_cfg["activation"], K, l_info)
        }

def get_model_config(model_name: str):
    loader = ModelLoader(model_name)
    
    # Return structure compatible with simulator.py
    config = {
        "input_shape": loader.info["input_shape"],
        "layers": loader.layers,
        "k_sizes": {l["name"]: l["K"] for l in loader.layers},
        "layer_shapes": {l["name"]: {"P": l["P"], "Q": l["Q"]} for l in loader.layers},
        "quant_cfgs": {l["name"]: l["quant"] for l in loader.layers}
    }
    return config
