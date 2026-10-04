"""
generate_unpacked_c.py
----------------------
Generates standard unpacked int8_t arrays for multi-precision evaluation on FPGA.
Values are quantized to their respective MQF bit-widths but stored in standard 8-bit C types.
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
FW_DIR      = os.path.join(PROJECT_ROOT, "quantization_framework")

for p in [PROJECT_ROOT, FW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Model registry (mirrors model_loader.py) ─────────────────────────────────
MODEL_REGISTRY = {
    "vgg11": {
        "pth":         os.path.join(PROJECT_ROOT, "models", "vgg11_cifar10_fp32.pth"),
        "json":        os.path.join(FW_DIR, "configs", "vgg11_bn_config_2_4_8_weight.json"),
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

def load_model(model_name: str, checkpoint_override: str = None):
    info = MODEL_REGISTRY[model_name]
    cls  = info["class"]

    from quantization_framework.models.vgg import VGG, vgg11_bn
    from quantization_framework.models.resnet import ResNet, ResNet18, BasicBlock
    import types
    import __main__

    __main__.VGG                  = VGG
    __main__.vgg11_bn             = vgg11_bn
    __main__.ResNet               = ResNet
    __main__.ResNet18             = ResNet18
    __main__.BasicBlock           = BasicBlock

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

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        checkpoint = model.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        if cls == "vgg11_bn":
            model = vgg11_bn(num_classes=10)
        else:
            model = ResNet18(num_classes=10)
        checkpoint = model.state_dict()
    else:
        if cls == "vgg11_bn":
            model = vgg11_bn(num_classes=10)
        elif cls == "ResNet18":
            model = ResNet18(num_classes=10, cifar=True)
        else:
            model = eval(cls)(num_classes=10)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, checkpoint


def expand_bitwidths(val, P: int) -> List[int]:
    """Broadcast scalar/list bit-widths to flat P-length arrays."""
    if isinstance(val, (int, float)):
        return [int(val)] * P
    if isinstance(val, list):
        if len(val) == P:
            return [int(v) for v in val]
        max_b = int(max(val))
        return [max_b] * P
    return [8] * P


def extract_layers(model: nn.Module, input_shape: Tuple[int, int, int],
                   quant_json: dict, checkpoint: dict) -> List[dict]:
    layers = []
    C, H, W = input_shape

    modules = list(model.named_modules())
    total_ops = sum(1 for name, module in modules if isinstance(module, (nn.Conv2d, nn.Linear)))
    op_idx = 0

    for i, (name, module) in enumerate(modules):
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
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
                
                if not is_final_layer:
                    for j in range(i + 1, len(modules)):
                        next_name, next_mod = modules[j]
                        if isinstance(next_mod, (nn.Conv2d, nn.Linear)):
                            next_i_scale_key = f"{next_name}.input_scale"
                            if next_i_scale_key in state_dict:
                                o_scale = state_dict[next_i_scale_key].item()
                            break

                w_scale = state_dict[w_scale_key].numpy()
                
                if abs(i_scale - 1.0) < 1e-6: i_scale = 1.0 / 127.0
                if abs(o_scale - 1.0) < 1e-6 and not is_final_layer: o_scale = 1.0 / 127.0

                bn_module = None
                for j in range(i + 1, len(modules)):
                    next_name, next_mod = modules[j]
                    if isinstance(next_mod, nn.BatchNorm2d):
                        bn_module = next_mod
                        break
                    if isinstance(next_mod, (nn.Conv2d, nn.Linear)): break
                
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
                
                b_conv = module.bias.data.numpy() if getattr(module, 'bias', None) is not None else np.zeros(C_out)
                b_folded = b_conv * gammas + betas
                bias_data = [int(round(float(b / o_scale) / float(s))) if float(s) != 0 else 0 for b, s in zip(b_folded, scale_arr)]
            else:
                bn_module = None
                for j in range(i + 1, len(modules)):
                    next_name, next_mod = modules[j]
                    if isinstance(next_mod, nn.BatchNorm2d):
                        bn_module = next_mod
                        break
                    if isinstance(next_mod, (nn.Conv2d, nn.Linear)): break
                
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
                
                b_conv = module.bias.data.numpy() if getattr(module, 'bias', None) is not None else np.zeros(C_out)
                b_folded = b_conv * gammas + betas
                
                w_max_per_ch = np.max(np.abs(w_folded), axis=(1, 2, 3))
                scale_arr = [max(m / 127.0, 1e-9) for m in w_max_per_ch]
                
                bias_data = [int(round(b / s)) for b, s in zip(b_folded, scale_arr)]

            layer_meta = {
                "name": name, "type": "conv", "module": module,
                "K": K, "P": C_out, "Q": H_out * W_out,
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
                
                if not is_final_layer:
                    for j in range(i + 1, len(modules)):
                        next_name, next_mod = modules[j]
                        if isinstance(next_mod, (nn.Conv2d, nn.Linear)):
                            next_i_scale_key = f"{next_name}.input_scale"
                            if next_i_scale_key in state_dict:
                                o_scale = state_dict[next_i_scale_key].item()
                            break

                w_scale = state_dict[w_scale_key].numpy()
                
                if abs(i_scale - 1.0) < 1e-6: i_scale = 1.0 / 127.0
                if abs(o_scale - 1.0) < 1e-6 and not is_final_layer: o_scale = 1.0 / 127.0

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
                "name": name, "type": "fc", "module": module,
                "K": K, "P": module.out_features, "Q": 1,
                "bias": bias_data, "scale": scale_arr
            }

        if name in quant_json:
            q_cfg = quant_json[name]
        else:
            q_cfg = {"weight": 8, "activation": 8}

        if isinstance(q_cfg, dict):
            w_bit_val = q_cfg.get("weight", 8)
            a_bit_val = q_cfg.get("activation", 8)
        else:
            w_bit_val = q_cfg
            a_bit_val = q_cfg

        layer_meta["w_bits"] = expand_bitwidths(w_bit_val, layer_meta["P"])
        layer_meta["a_bits"] = expand_bitwidths(a_bit_val, layer_meta["P"])
        layers.append(layer_meta)

    return layers


def build_unpacked_weights(layer: dict, state_dict: dict) -> List[int]:
    """Quantize weights according to the layer's per-filter bitwidth configuration and return as a flat list."""
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

    unpacked_words = []
    
    # Iterate through each output channel (P)
    for p in range(w_flat.shape[0]):
        w_vec = w_flat[p].numpy()
        bits = layer["w_bits"][p] # Per-filter bitwidth
        
        w_zp_key = f"{layer['name']}.weight._zeropoint"
        zp = state_dict[w_zp_key][p].item() if w_zp_key in state_dict else 0
        
        if is_already_quantized:
            w_int_vec = w_vec - zp
        elif w_scale_key in state_dict:
            w_scale = state_dict[w_scale_key][p].item()
            w_int_vec = np.round(w_vec / w_scale)
        else:
            w_int_vec = np.round((w_vec / (w_max + 1e-8)) * 127.0)
            
        # Clamp to specific bitwidth limits
        max_int = (1 << (bits - 1)) - 1
        min_int = -(1 << (bits - 1))
        
        for raw_int in w_int_vec:
            q = max(min_int, min(max_int, int(raw_int)))
            unpacked_words.append(q)
            
    return unpacked_words


def c_safe_name(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("-", "_")


def write_header(model_name: str, layers: List[dict], out_path: str, is_baseline: bool = False):
    guard_suffix = "_BASELINE_H" if is_baseline else "_UNPACKED_H"
    guard = f"PACKED_WEIGHTS_{model_name.upper()}{guard_suffix}"
    lines = []

    lines.append(f"/* Auto-generated by generate_unpacked_c.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} */")
    lines.append(f"/* Model: {model_name}  |  UNPACKED MULTI-PRECISION */")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    for layer in layers:
        lname = layer["name"]
        cname = c_safe_name(lname)
        n_elements = layer["P"] * layer["K"]

        lines.append(f"/* -- {lname}  (type={layer['type']}, P={layer['P']}, K={layer['K']}) */")
        lines.append(f"#define {cname.upper()}_N_ELEMENTS {n_elements}")
        lines.append(f"extern const int8_t {cname}_weights[{n_elements}];")
        lines.append(f"extern const float {cname}_scale[{layer['P']}];")
        lines.append(f"extern const int32_t {cname}_bias[{layer['P']}];")
        lines.append(f"extern const uint8_t {cname}_out_act_bits[{layer['P']}];")
        lines.append("")

    lines.append(f"#endif /* {guard} */")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_source(model_name: str, layers: List[dict], packed_data: Dict[str, List[int]], out_path: str, header_name: str):
    lines = []
    lines.append(f"/* Auto-generated by generate_unpacked_c.py — {datetime.now().strftime('%Y-%m-%d %H:%M')} */")
    lines.append(f"/* Model: {model_name}  |  UNPACKED MULTI-PRECISION */")
    lines.append(f'#include "{header_name}"')
    lines.append("")

    for layer in layers:
        lname = layer["name"]
        cname = c_safe_name(lname)
        words_data = packed_data.get(lname, [])
        n_elements = len(words_data)

        lines.append(f"/* -- {lname} -- */")

        lines.append(f"const int8_t {cname}_weights[{n_elements}] = {{")
        row = []
        for i, val in enumerate(words_data):
            row.append(str(val))
            if len(row) == 16 or i == n_elements - 1:
                lines.append("    " + ", ".join(row) + ("," if i < n_elements - 1 else ""))
                row = []
        lines.append("};")

        bias_list = layer.get("bias", [0] * layer["P"])
        lines.append(f"const int32_t {cname}_bias[{len(bias_list)}] = {{")
        row = []
        for i, val in enumerate(bias_list):
            row.append(str(int(val)))
            if len(row) == 8 or i == len(bias_list) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(bias_list) - 1 else ""))
                row = []
        lines.append("};")

        scale_list = layer.get("scale", [1.0] * layer["P"])
        lines.append(f"const float {cname}_scale[{len(scale_list)}] = {{")
        row = []
        for i, val in enumerate(scale_list):
            row.append(f"{float(val):.6f}f")
            if len(row) == 8 or i == len(scale_list) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(scale_list) - 1 else ""))
                row = []
        lines.append("};")

        a_bits_out = layer.get("a_bits", [8] * layer["P"])
        lines.append(f"const uint8_t {cname}_out_act_bits[{len(a_bits_out)}] = {{")
        row = []
        for i, val in enumerate(a_bits_out):
            row.append(str(int(val)))
            if len(row) == 8 or i == len(a_bits_out) - 1:
                lines.append("    " + ", ".join(row) + ("," if i < len(a_bits_out) - 1 else ""))
                row = []
        lines.append("};")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="vgg11")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    is_baseline = args.baseline
    info = MODEL_REGISTRY[model_name]

    print(f"\n[1/4] Loading model architecture: {model_name}")
    model, checkpoint = load_model(model_name, args.checkpoint)

    print(f"[2/4] Loading MQF JSON config: {os.path.basename(info['json'])}")
    with open(info["json"]) as f:
        quant_json_raw = json.load(f)
        
    if "config" in quant_json_raw:
        quant_json = quant_json_raw["config"]
    else:
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

    print(f"[3/4] Extracting layers and building unpacked weights...")
    layers = extract_layers(model, info["input_shape"], quant_json, checkpoint)
    if is_baseline:
        for layer in layers:
            layer["w_bits"] = [8] * layer["P"]
            layer["a_bits"] = [8] * layer["P"]

    packed_data: Dict[str, List[int]] = {}
    for layer in layers:
        words_data = build_unpacked_weights(layer, checkpoint)
        packed_data[layer["name"]] = words_data

    print(f"[4/4] Writing unpacked C files...")
    out_dir = os.path.abspath(os.path.join(THIS_DIR, "..", "generated"))
    os.makedirs(out_dir, exist_ok=True)

    suffix = "_baseline_unpacked" if is_baseline else "_mqf_unpacked"
    header_name = f"{model_name}{suffix}.h"
    source_name = f"{model_name}{suffix}.cc"
    header_path = os.path.join(out_dir, header_name)
    source_path = os.path.join(out_dir, source_name)

    write_header(model_name, layers, header_path, is_baseline=is_baseline)
    write_source(model_name, layers, packed_data, source_path, header_name)

    print(f"\nDone. Files written to: risc_v_backend/generated/{header_name}")

if __name__ == "__main__":
    main()
