"""
generate_packed_c.py
--------------------
MQF hardware packing script.
Parses trained PyTorch weights and JSON bit-width configurations to generate 
static C arrays packed via the Safe-FFD algorithm for RISC-V SWAR execution.
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

# ── Path setup: reach project root and systolic_sim ──────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SIM_DIR     = os.path.join(PROJECT_ROOT, "systolic_sim")
FW_DIR      = os.path.join(PROJECT_ROOT, "quantization_framework")

for p in [PROJECT_ROOT, SIM_DIR, FW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from packing import pack_safe_ffd, PackingResult, RegisterWord, OpdField

# ── Model registry (mirrors model_loader.py) ─────────────────────────────────
MODEL_REGISTRY = {
    "alexnet": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "alexnet_fashionmnist_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "alexnet_fashionmnist_channel_config_2_4_8_weight.json"),
        "input_shape": (1, 227, 227),
        "class":       "AlexNet",
    },
    "vgg11": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "vgg11_cifar10_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "vgg11_bn_cifar10_channel_config_2_4_8_weight.json"),
        "input_shape": (3, 32, 32),
        "class":       "vgg11_bn",
    },
    "resnet18": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "resnet18_cifar10_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "resnet_cifar10_channel_config_2_4_8_weight.json"),
        "input_shape": (3, 32, 32),
        "class":       "ResNet18",
    },
    "swin": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "swin_cifar100_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "swin_cifar100_channel_config_2_4_8_weight.json"),
        "input_shape": (3, 224, 224),
        "class":       "SwinTransformer",
    },
    "mobilenet": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "mobilenet_cifar10_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "mobilenet_cifar10_channel_config_2_4_8_weight.json"),
        "input_shape": (3, 32, 32),
        "class":       "MobileNetV2",
    },
}

REGISTER_WIDTH = 32   # R in bits — matches the systolic simulator default

# Max K slots per layer — layers above this are packed with baseline (d=1)
# to avoid O(n^2) Safe-FFD runtime on large FC layers.
MAX_K_FOR_SAFE_FFD = 5000


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(model_name: str, checkpoint_override: str = None):
    """Load trained architecture and weights, injecting aliases for unpickling."""
    info = MODEL_REGISTRY[model_name]
    cls  = info["class"]

    # ── Import the model classes and inject into __main__ ────────────────
    # Alias training classes for torch.load compatibility
    from quantization_framework.models.alexnet import AlexNet
    from quantization_framework.models.vgg import VGG, vgg11_bn
    from quantization_framework.models.resnet import ResNet, ResNet18, BasicBlock
    import types
    import __main__

    # Inject all class names the .pth files may have been pickled with
    __main__.fasion_mnist_alexnet = AlexNet   # typo in original training code
    __main__.AlexNet              = AlexNet
    __main__.VGG                  = VGG
    __main__.vgg11_bn             = vgg11_bn
    __main__.ResNet               = ResNet
    __main__.ResNet18             = ResNet18
    __main__.BasicBlock           = BasicBlock

    # Also inject into sys.modules paths used during pickling
    fake_resnet = types.ModuleType("models.resnet")
    fake_resnet.ResNet     = ResNet
    fake_resnet.ResNet18   = ResNet18
    fake_resnet.BasicBlock = BasicBlock
    sys.modules.setdefault("models", types.ModuleType("models"))
    sys.modules["models.resnet"] = fake_resnet

    fake_vgg = types.ModuleType("models.vgg")
    fake_vgg.VGG      = VGG
    fake_vgg.vgg11_bn = vgg11_bn
    sys.modules["models.vgg"] = fake_vgg

    pth_path = checkpoint_override if checkpoint_override else info["pth"]
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f".pth not found: {pth_path}")

    print(f"      Loading .pth: {os.path.basename(pth_path)}")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    # The checkpoint may be the full model object or a state_dict
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        checkpoint = model.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Wrapped checkpoint
        if cls == "AlexNet":
            model = AlexNet(num_classes=10)
        elif cls == "vgg11_bn":
            model = vgg11_bn(num_classes=10)
        else:
            model = ResNet18(num_classes=10)
        checkpoint = model.state_dict()
    else:
        if cls == "SwinTransformer":
            from quantization_framework.models.swin import swin_tiny_patch4_window7_224
            model = swin_tiny_patch4_window7_224()
        elif cls == "MobileNetV2":
            from quantization_framework.models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=10)
        elif cls == "ResNet18":
            model = ResNet18(num_classes=10, cifar=True)
        else:
            model = eval(cls)(num_classes=10)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, checkpoint


def expand_bitwidths(val, K: int, layer_meta: dict) -> List[int]:
    """Broadcast scalar/list bit-widths to flat K-length arrays."""
    if isinstance(val, (int, float)):
        return [int(val)] * K

    if isinstance(val, list):
        if len(val) == K:
            return [int(v) for v in val]
        
        # If it's a depthwise layer and we are expanding bitwidths per channel
        # the list might contain C_out elements, but K is just kH*kW. 
        # In this script, packing happens layer-wise so we must pick a uniform bitwidth 
        # for the K dimension. Danial's packing algorithm uses the max bits across all channels 
        # if the bitwidths differ per output channel.
        # But wait, we just extract the maximum bitwidth across the whole array to be safe.
        max_b = int(max(val))
        return [max_b] * K

    return [8] * K


def extract_layers(model: nn.Module, input_shape: Tuple[int, int, int],
                   quant_json: dict, checkpoint: dict) -> List[dict]:
    """Extract topology and parse layer parameters (K, scaling, bit-widths)."""
    layers = []
    C, H, W = input_shape

    modules = list(model.named_modules())
    # Count total Conv/Linear ops to identify the final layer
    total_ops = sum(1 for name, module in modules if isinstance(module, (nn.Conv2d, nn.Linear)))
    op_idx = 0

    for i, (name, module) in enumerate(modules):
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            # Track spatial dims through pooling
            if isinstance(module, nn.MaxPool2d):
                kh = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                st = module.stride      if isinstance(module.stride, int)      else module.stride[0]
                pa = module.padding     if isinstance(module.padding, int)     else module.padding[0]
                H = (H + 2*pa - kh) // st + 1
                W = (W + 2*pa - kh) // st + 1
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                out = module.output_size
                H, W = (out, out) if isinstance(out, int) else out
            continue

        op_idx += 1
        is_final_layer = (op_idx == total_ops)

        if isinstance(module, nn.Conv2d):
            C_out = module.out_channels
            C_in  = module.in_channels
            kH, kW = module.kernel_size
            stride  = module.stride[0]
            padding = module.padding[0]
            groups  = module.groups
            H_out = (H + 2*padding - kH) // stride + 1
            W_out = (W + 2*padding - kW) // stride + 1
            
            is_depthwise = (groups == C_in and C_in == C_out)
            K = (C_in // groups) * kH * kW
            
            state_dict = checkpoint
            i_scale_key = f"{name}.input_scale"
            o_scale_key = f"{name}.output_scale"
            w_scale_key = f"{name}.weight._scale"
            
            if i_scale_key in state_dict and o_scale_key in state_dict and w_scale_key in state_dict:
                i_scale = state_dict[i_scale_key].item()
                o_scale = state_dict[o_scale_key].item()
                w_scale = state_dict[w_scale_key].numpy()
                
                # Map float activations [-1, 1] to int8 [-128, 127]
                if abs(i_scale - 1.0) < 1e-6:
                    i_scale = 1.0 / 127.0
                if abs(o_scale - 1.0) < 1e-6 and not is_final_layer:
                    o_scale = 1.0 / 127.0

                import numpy as np
                bn_module = None
                for j in range(i + 1, len(modules)):
                    next_name, next_mod = modules[j]
                    if isinstance(next_mod, nn.BatchNorm2d):
                        bn_module = next_mod
                        break
                    if isinstance(next_mod, (nn.Conv2d, nn.Linear)):
                        break
                
                gammas = np.ones(C_out)
                betas = np.zeros(C_out)
                if bn_module is not None:
                    rm = bn_module.running_mean.detach().numpy()
                    rv = bn_module.running_var.detach().numpy()
                    bn_w = bn_module.weight.detach().numpy()
                    bn_b = bn_module.bias.detach().numpy()
                    eps = bn_module.eps
                    gammas = bn_w / np.sqrt(rv + eps)
                    betas = bn_b - rm * gammas

                scale_arr = [(i_scale * w * g) / o_scale for w, g in zip(w_scale, gammas)]
                
                if getattr(module, 'bias', None) is not None:
                    b_conv = module.bias.data.numpy()
                else:
                    b_conv = np.zeros(C_out)
                    
                b_folded = b_conv * gammas + betas
                bias_data = [int(round(float(b / o_scale) / float(s))) if float(s) != 0 else 0 for b, s in zip(b_folded, scale_arr)]
            else:
                import numpy as np
                bn_module = None
                for j in range(i + 1, len(modules)):
                    next_name, next_mod = modules[j]
                    if isinstance(next_mod, nn.BatchNorm2d):
                        bn_module = next_mod
                        break
                    if isinstance(next_mod, (nn.Conv2d, nn.Linear)):
                        break
                
                gammas = np.ones(C_out)
                betas = np.zeros(C_out)
                if bn_module is not None:
                    rm = bn_module.running_mean.detach().numpy()
                    rv = bn_module.running_var.detach().numpy()
                    bn_w = bn_module.weight.detach().numpy()
                    bn_b = bn_module.bias.detach().numpy()
                    eps = bn_module.eps
                    gammas = bn_w / np.sqrt(rv + eps)
                    betas = bn_b - rm * gammas
                
                w = module.weight.detach().numpy()
                w_folded = w * gammas[:, None, None, None]
                
                if getattr(module, 'bias', None) is not None:
                    b_conv = module.bias.data.numpy()
                else:
                    b_conv = np.zeros(C_out)
                
                b_folded = b_conv * gammas + betas
                
                w_max_per_ch = np.max(np.abs(w_folded), axis=(1, 2, 3))
                scale_arr = [max(m / 127.0, 1e-9) for m in w_max_per_ch]
                
                bias_data = [int(round(b / s)) for b, s in zip(b_folded, scale_arr)]

            layer_meta = {
                "name": name, "type": "conv",
                "module": module,
                "K": K, "P": C_out, "Q": H_out * W_out,
                "in_ch": C_in, "out_ch": C_out,
                "kernel": kH, "stride": stride, "padding": padding,
                "groups": groups, "is_depthwise": is_depthwise,
                "bias": bias_data, "scale": scale_arr
            }
            H, W, C = H_out, W_out, C_out
        else:  # Linear
            K = module.in_features
            
            state_dict = checkpoint
            i_scale_key = f"{name}.input_scale"
            o_scale_key = f"{name}.output_scale"
            w_scale_key = f"{name}.weight._scale"
            
            if i_scale_key in state_dict and o_scale_key in state_dict and w_scale_key in state_dict:
                i_scale = state_dict[i_scale_key].item()
                o_scale = state_dict[o_scale_key].item()
                w_scale = state_dict[w_scale_key].numpy()
                
                # Dynamic activation scaling fix
                if abs(i_scale - 1.0) < 1e-6:
                    i_scale = 1.0 / 127.0
                if abs(o_scale - 1.0) < 1e-6 and not is_final_layer:
                    o_scale = 1.0 / 127.0

                scale_arr = [(i_scale * w) / o_scale for w in w_scale]
                
                if getattr(module, 'bias', None) is not None:
                    bias_float = module.bias.data.numpy()
                    bias_data = [int(round(float(b / o_scale) / float(i_scale * w / o_scale))) for b, w in zip(bias_float, w_scale)]
                else:
                    bias_data = [0] * module.out_features
            else:
                w_max = float(module.weight.abs().max().item()) if module.weight is not None else 1.0
                scale_arr = [w_max / 127.0] * module.out_features
                bias_data = module.bias.data.numpy().tolist() if getattr(module, 'bias', None) is not None else [0] * module.out_features

            layer_meta = {
                "name": name, "type": "fc",
                "module": module,
                "K": K, "P": module.out_features, "Q": 1,
                "in_ch": K, "out_ch": module.out_features,
                "kernel": 1,
                "bias": bias_data, "scale": scale_arr
            }

        # Look up the JSON config for this layer
        if name in quant_json:
            q_cfg = quant_json[name]
        else:
            print(f"  [WARN] Layer '{name}' not in JSON config — using 8-bit default")
            q_cfg = {"weight": 8, "activation": 8}

        if isinstance(q_cfg, dict):
            w_bit_val = q_cfg.get("weight", 8)
            a_bit_val = q_cfg.get("activation", 8)
        else:
            w_bit_val = q_cfg
            a_bit_val = q_cfg

        layer_meta["w_bits"] = expand_bitwidths(w_bit_val, K, layer_meta)
        layer_meta["a_bits"] = expand_bitwidths(a_bit_val, K, layer_meta)
        layers.append(layer_meta)

    return layers


def quantize_weight(value: float, bits: int) -> int:
    """Symmetric uniform quantization float -> signed int."""
    if bits <= 0:
        return 0
    max_int = (1 << (bits - 1)) - 1
    min_int = -(1 << (bits - 1))
    # Scale: assume weights are normalised to roughly [-1, 1]
    # Use the observed max of the weight tensor for proper scaling
    scaled = value * max_int
    return int(max(min_int, min(max_int, round(scaled))))


def build_packed_words(layer: dict, packing_result: PackingResult, state_dict: dict) -> List[int]:
    """Pack quantized weights into R-bit register words according to Safe-FFD mapping."""
    w_data_key = f"{layer['name']}.weight._data"
    w_scale_key = f"{layer['name']}.weight._scale"
    
    if w_data_key in state_dict:
        weight_tensor = state_dict[w_data_key].float() # Already quantized to [-128, 127]
    else:
        weight_tensor = layer["module"].weight.data  # float tensor

    if layer["type"] == "conv":
        w_flat = weight_tensor.view(weight_tensor.shape[0], -1)
    else:  # Linear
        w_flat = weight_tensor

    is_already_quantized = w_data_key in state_dict
    w_max = float(weight_tensor.abs().max().item()) or 1.0

    packed_words = []
    # Iterate through each output channel (P)
    for p in range(w_flat.shape[0]):
        w_vec = w_flat[p].numpy()
        
        # Determine exact integer weights for this channel
        w_zp_key = f"{layer['name']}.weight._zeropoint"
        zp = state_dict[w_zp_key][p].item() if w_zp_key in state_dict else 0
        
        if is_already_quantized:
            w_int_vec = w_vec - zp
        elif w_scale_key in state_dict:
            w_scale = state_dict[w_scale_key][p].item()
            w_int_vec = np.round(w_vec / w_scale)
        else:
            w_int_vec = np.round((w_vec / (w_max + 1e-8)) * 127.0)
            
        for word in packing_result.words:
            word_val = 0
            for field in word.fields:
                raw_int = int(w_int_vec[field.slot_index]) if field.slot_index < len(w_int_vec) else 0
                
                # Clamp to bitwidth limits
                max_int = (1 << (field.weight_bits - 1)) - 1
                min_int = -(1 << (field.weight_bits - 1))
                q = max(min_int, min(max_int, raw_int))
                
                mask = (1 << field.weight_bits) - 1
                word_val |= (q & mask) << field.bit_offset
            packed_words.append(word_val)
            
    return packed_words


def c_type_for_register(r: int) -> str:
    """Return the smallest C unsigned integer type that holds r bits."""
    if r <= 8:
        return "uint8_t"
    elif r <= 16:
        return "uint16_t"
    else:
        return "uint32_t"


def c_safe_name(layer_name: str) -> str:
    """Convert 'features.0' → 'features_0' (valid C identifier)."""
    return layer_name.replace(".", "_").replace("-", "_")


# ── Header file writer ────────────────────────────────────────────────────────

def write_header(model_name: str, layers: List[dict],
                 packing_results: Dict[str, PackingResult],
                 out_path: str, is_baseline: bool = False):
    """Generate C header with extern declarations and LUT metadata."""
    guard_suffix = "_BASELINE_H" if is_baseline else "_H"
    guard = f"PACKED_WEIGHTS_{model_name.upper()}{guard_suffix}"
    ctype = c_type_for_register(REGISTER_WIDTH)
    lines = []

    lines.append(f"/* Auto-generated by generate_packed_c.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} */")
    lines.append(f"/* Model: {model_name}  |  Register width R = {REGISTER_WIDTH} bits */")
    lines.append(f"/* DO NOT EDIT — regenerate with: python generate_packed_c.py --model {model_name} */")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    for layer in layers:
        lname = layer["name"]
        cname = c_safe_name(lname)
        result = packing_results.get(lname)
        if result is None:
            continue

        n_words = result.n_words * layer["P"]
        max_d   = max((w.d for w in result.words), default=1)
        # Determine the dominant bit-width for this layer
        all_bw  = [f.weight_bits for w in result.words for f in w.fields]
        dom_bw  = max(set(all_bw), key=all_bw.count) if all_bw else 8

        words_per_filter = result.n_words
        meta_size = words_per_filter * max_d

        lines.append(f"/* -- {lname}  (type={layer['type']}, "
                     f"K={layer['K']}, n_words={n_words}, max_d={max_d}, "
                     f"dominant_bits={dom_bw}) */")
        lines.append(f"#define {cname.upper()}_N_WORDS   {n_words}")
        lines.append(f"#define {cname.upper()}_WORDS_PER_FILTER {words_per_filter}")
        lines.append(f"#define {cname.upper()}_MAX_D     {max_d}")
        lines.append(f"#define {cname.upper()}_META_SIZE {meta_size}")
        lines.append(f"#define {cname.upper()}_REG_BITS  {REGISTER_WIDTH}")
        lines.append(f"extern const {ctype} {cname}_weights[{n_words}];")
        lines.append(f"extern const float {cname}_scale[{layer['out_ch']}];")
        lines.append(f"extern const int32_t {cname}_bias[{layer['out_ch']}];")
        lines.append(f"extern const uint8_t {cname}_out_act_bits[{layer['out_ch']}];")

        lines.append(f"extern const uint8_t {cname}_pos[{meta_size}];")
        lines.append(f"extern const uint8_t {cname}_mask[{meta_size}];")
        lines.append(f"extern const uint16_t {cname}_slots[{meta_size}];")
        lines.append("")

    lines.append(f"#endif /* {guard} */")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  >> Header  : {os.path.relpath(out_path)}")


# ── Source file writer ────────────────────────────────────────────────────────

def write_source(model_name: str, layers: List[dict],
                 packing_results: Dict[str, PackingResult],
                 packed_data: Dict[str, List[int]],
                 out_path: str, header_name: str):
    """Generate C source containing static packed data arrays."""
    ctype = c_type_for_register(REGISTER_WIDTH)
    lines = []

    lines.append(f"/* Auto-generated by generate_packed_c.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} */")
    lines.append(f"/* Model: {model_name}  |  Register width R = {REGISTER_WIDTH} bits */")
    lines.append(f"/* Compile: riscv-none-embed-gcc -march=rv32i -mabi=ilp32 -O2 */")
    lines.append("")
    lines.append(f'#include "{header_name}"')
    lines.append("")

    for layer in layers:
        lname = layer["name"]
        cname = c_safe_name(lname)
        result = packing_results.get(lname)
        if result is None:
            continue

        words_data = packed_data.get(lname, [])
        n_words    = len(words_data)
        max_d      = max((w.d for w in result.words), default=1)

        repr_word = max(result.words, key=lambda w: w.d) if result.words else None
        pos_vals  = [f.bit_offset    for f in repr_word.fields] if repr_word else [0]
        mask_vals = [(1 << f.weight_bits) - 1 for f in repr_word.fields] if repr_word else [0xFF]

        lines.append(f"/* -- {lname} -- */")

        # Packed weight array — 8 values per line for readability
        lines.append(f"const {ctype} {cname}_weights[{n_words}] = {{")
        hex_width = REGISTER_WIDTH // 4  # nibbles: 4 bits per hex digit
        fmt = f"{{}}"
        row = []
        for i, val in enumerate(words_data):
            row.append(fmt.format(val & ((1 << REGISTER_WIDTH) - 1)))
            if len(row) == 8 or i == n_words - 1:
                lines.append("    " + ", ".join(row) + ("," if i < n_words - 1 else ""))
                row = []
        lines.append("};")

        # Bias array
        bias_list = layer.get("bias", [0] * layer["out_ch"])
        lines.append(f"const int32_t {cname}_bias[{len(bias_list)}] = {{")
        row = []
        for i, val in enumerate(bias_list):
            row.append(str(int(val)))
            if len(row) == 8 or i == len(bias_list) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(bias_list) - 1 else ""))
                row = []
        lines.append("};")

        scale_list = layer.get("scale", [1.0] * layer["out_ch"])
        lines.append(f"const float {cname}_scale[{len(scale_list)}] = {{")
        row = []
        for i, val in enumerate(scale_list):
            row.append(f"{float(val):.6f}f")
            if len(row) == 8 or i == len(scale_list) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(scale_list) - 1 else ""))
                row = []
        lines.append("};")

        # Activation bits array
        a_bits_list = layer.get("a_bits", [8])
        if isinstance(a_bits_list, list):
            a_bits_val = a_bits_list[0]
        else:
            a_bits_val = a_bits_list
            
        a_bits_out = [a_bits_val] * layer["out_ch"]
            
        lines.append(f"const uint8_t {cname}_out_act_bits[{layer['out_ch']}] = {{")
        row = []
        for i, val in enumerate(a_bits_out):
            row.append(str(int(val)))
            if len(row) == 8 or i == len(a_bits_out) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(a_bits_out) - 1 else ""))
                row = []
        lines.append("};")

        pos_array = []
        mask_array = []
        slots_array = []
        for word in result.words:
            for k in range(max_d):
                if k < len(word.fields):
                    f = word.fields[k]
                    pos_array.append(f.bit_offset)
                    mask_array.append((1 << f.weight_bits) - 1)
                    slots_array.append(f.slot_index)
                else:
                    pos_array.append(0)
                    mask_array.append(0)
                    slots_array.append(0)
        
        meta_size = len(pos_array)
        lines.append(f"const uint8_t {cname}_pos[{meta_size}]  = {{")
        row = []
        for i, val in enumerate(pos_array):
            row.append(str(val))
            if len(row) == 8 or i == len(pos_array) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(pos_array) - 1 else ""))
                row = []
        lines.append("};")

        lines.append(f"const uint8_t {cname}_mask[{meta_size}] = {{")
        row = []
        for i, val in enumerate(mask_array):
            row.append(hex(val))
            if len(row) == 8 or i == len(mask_array) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(mask_array) - 1 else ""))
                row = []
        lines.append("};")

        lines.append(f"const uint16_t {cname}_slots[{meta_size}] = {{")
        row = []
        for i, val in enumerate(slots_array):
            row.append(str(val))
            if len(row) == 8 or i == len(slots_array) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(slots_array) - 1 else ""))
                row = []
        lines.append("};")

        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  >> Source  : {os.path.relpath(out_path)}")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(model_name: str, layers: List[dict],
                  packing_results: Dict[str, PackingResult]):
    print(f"\n{'='*70}")
    print(f"  Packing Summary — {model_name.upper()}  (R={REGISTER_WIDTH} bits)")
    print(f"{'='*70}")
    print(f"  {'Layer':<30} {'Type':>5} {'K':>6} {'Words':>6} {'avg_d':>6} {'C type':>10}")
    print(f"  {'-'*65}")
    ctype = c_type_for_register(REGISTER_WIDTH)
    for layer in layers:
        lname = layer["name"]
        result = packing_results.get(lname)
        if result is None:
            continue
        avg_d = result.avg_d
        print(f"  {lname:<30} {layer['type']:>5} {layer['K']:>6} "
              f"{result.n_words:>6} {avg_d:>6.2f} {ctype:>10}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MQF config + .pth weights to typed C arrays for RISC-V SWAR")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="resnet18", help="Model to convert")
    parser.add_argument("--reg_width", type=int, default=REGISTER_WIDTH,
                        help=f"Register width in bits (default: {REGISTER_WIDTH})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional override for the model .pth checkpoint (useful for QAT weights)")
    parser.add_argument("--baseline", action="store_true",
                        help="Generate baseline uniform 8-bit weights (no packing)")
    args = parser.parse_args()

    model_name = args.model
    R = args.reg_width
    is_baseline = args.baseline
    checkpoint_override = args.checkpoint
    info = MODEL_REGISTRY[model_name]

    print(f"\n[1/5] Loading model architecture: {model_name}")
    model, checkpoint = load_model(model_name, checkpoint_override)

    print(f"[2/5] Loading MQF JSON config: {os.path.basename(info['json'])}")
    with open(info["json"]) as f:
        quant_json_raw = json.load(f)
        
    # Support both old format {"config": {"layer": {"weight": 8, "activation": 8}}}
    # and new format where weights and activations are in separate files.
    if "config" in quant_json_raw:
        quant_json = quant_json_raw["config"]
    else:
        # It's a weight-only config file. Try to load the activation counterpart.
        w_json = quant_json_raw
        a_path = info["json"].replace("_weight.json", "_activation.json")
        if os.path.exists(a_path):
            with open(a_path) as f:
                a_json = json.load(f)
        else:
            a_json = {}
            
        quant_json = {}
        for k in w_json.keys():
            quant_json[k] = {
                "weight": w_json[k],
                "activation": a_json.get(k, 8)
            }

    print(f"[3/5] Extracting layers and expanding bit-widths (K-length lists)...")
    layers = extract_layers(model, info["input_shape"], quant_json, checkpoint)
    if is_baseline:
        for layer in layers:
            layer["w_bits"] = [8] * len(layer["w_bits"])
            layer["a_bits"] = [8] * len(layer["a_bits"])
    print(f"      Found {len(layers)} weight-bearing layers")

    print(f"[4/5] Running Safe-FFD packing (R={R} bits)...")
    packing_results: Dict[str, PackingResult] = {}
    packed_data:     Dict[str, List[int]]     = {}

    for layer in layers:
        lname = layer["name"]
        channels = list(zip(layer["w_bits"], layer["a_bits"]))

        # Use Safe-FFD only for manageable layer sizes.
        # Large FC layers (K > MAX_K_FOR_SAFE_FFD) or uniform 8-bit layers (which can't pack under R=16 due to overflow)
        # fall back to baseline (d=1) to avoid O(n^2) runtime.
        if layer["K"] > MAX_K_FOR_SAFE_FFD or all(w == 8 for w, a in channels):
            from packing import pack_baseline
            result = pack_baseline(channels, r=R, layer_name=lname)
        else:
            result = pack_safe_ffd(channels, r=R, layer_name=lname)

        packing_results[lname] = result

        # Build packed word values from actual weights
        words_data = build_packed_words(layer, result, checkpoint)
        packed_data[lname] = words_data

    print_summary(model_name, layers, packing_results)

    print(f"[5/5] Writing C files...")
    out_dir = os.path.abspath(os.path.join(THIS_DIR, "..", "generated"))
    os.makedirs(out_dir, exist_ok=True)

    suffix = "_baseline_weights" if is_baseline else "_packed_weights"
    header_name = f"{model_name}{suffix}.h"
    source_name = f"{model_name}{suffix}.cc"
    header_path = os.path.join(out_dir, header_name)
    source_path = os.path.join(out_dir, source_name)

    write_header(model_name, layers, packing_results, header_path, is_baseline=is_baseline)
    write_source(model_name, layers, packing_results, packed_data,
                 source_path, header_name)

    print(f"\nDone. Files written to: risc_v_backend/generated/")
    print(f"  Compile check (PC):    gcc -std=c99 -I generated/ -c {source_name}")
    print(f"  RISC-V cross-compile:  riscv-none-embed-gcc -march=rv32i -mabi=ilp32 "
          f"-I generated/ -c {source_name}")


if __name__ == "__main__":
    main()
