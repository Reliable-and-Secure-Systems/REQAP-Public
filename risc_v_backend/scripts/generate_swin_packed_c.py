import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SIM_DIR = os.path.join(PROJECT_ROOT, "systolic_sim")
FW_DIR = os.path.join(PROJECT_ROOT, "quantization_framework_ViT")

for p in [PROJECT_ROOT, SIM_DIR, FW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from packing import pack_safe_ffd, PackingResult
from quantization_framework_ViT.models.model_loaders import load_model

REGISTER_WIDTH = 16
MAX_K_FOR_SAFE_FFD = 5000

def expand_bitwidths(val, K: int) -> List[int]:
    if isinstance(val, (int, float)):
        return [int(val)] * K
    if isinstance(val, list):
        if len(val) == K:
            return [int(v) for v in val]
        return [int(val[i % len(val)]) for i in range(K)]
    return [8] * K

def extract_swin_layers(model: nn.Module, quant_json: dict) -> List[dict]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            K = module.in_features
            P = module.out_features
            
            # Use max to approximate scale if no PTQ stats exist in dict
            w_max = float(module.weight.abs().max().item()) if module.weight is not None else 1.0
            scale_arr = [max(w_max / 127.0, 1e-9)] * P
            bias_data = module.bias.data.numpy().tolist() if module.bias is not None else [0] * P

            layer_meta = {
                "name": name, "type": "fc",
                "module": module,
                "K": K, "P": P, "Q": 1,
                "bias": bias_data, "scale": scale_arr
            }

            if name in quant_json:
                q_cfg = quant_json[name]
            else:
                q_cfg = 8

            layer_meta["w_bits"] = expand_bitwidths(q_cfg, K)
            layer_meta["a_bits"] = expand_bitwidths(q_cfg, K)
            layers.append(layer_meta)
    return layers

def build_packed_words(layer: dict, packing_result: PackingResult) -> List[int]:
    w_flat = layer["module"].weight.data.float()
    w_max = float(w_flat.abs().max().item()) or 1.0

    packed_words = []
    for p in range(w_flat.shape[0]):
        w_vec = w_flat[p].numpy()
        w_int_vec = np.round((w_vec / (w_max + 1e-8)) * 127.0)
            
        for word in packing_result.words:
            word_val = 0
            for field in word.fields:
                raw_int = int(w_int_vec[field.slot_index]) if field.slot_index < len(w_int_vec) else 0
                max_int = (1 << (field.weight_bits - 1)) - 1
                min_int = -(1 << (field.weight_bits - 1))
                q = max(min_int, min(max_int, raw_int))
                mask = (1 << field.weight_bits) - 1
                word_val |= (q & mask) << field.bit_offset
            packed_words.append(word_val)
    return packed_words

def c_safe_name(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("-", "_")

def write_files(layers: List[dict], packing_results: Dict[str, PackingResult], packed_data: Dict[str, List[int]], out_dir: str):
    header_path = os.path.join(out_dir, "swin_packed_weights.h")
    source_path = os.path.join(out_dir, "swin_packed_weights.cc")
    
    h_lines = ["#ifndef SWIN_PACKED_H", "#define SWIN_PACKED_H", "#include <stdint.h>"]
    c_lines = ['#include "swin_packed_weights.h"']

    for layer in layers:
        lname = layer["name"]
        cname = c_safe_name(lname)
        result = packing_results.get(lname)
        if not result: continue

        n_words = result.n_words * layer["P"]
        max_d = max((w.d for w in result.words), default=1)
        meta_size = result.n_words * max_d
        
        h_lines.append(f"#define {cname.upper()}_MAX_D {max_d}")
        h_lines.append(f"#define {cname.upper()}_N_WORDS {n_words}")
        h_lines.append(f"#define {cname.upper()}_WORDS_PER_FILTER {result.n_words}")
        h_lines.append(f"extern const uint16_t {cname}_weights[{n_words}];")
        h_lines.append(f"extern const int32_t {cname}_bias[{layer['P']}];")
        h_lines.append(f"extern const float {cname}_scale[{layer['P']}];")
        if max_d > 1:
            h_lines.append(f"extern const uint8_t {cname}_pos[{meta_size}];")
            h_lines.append(f"extern const uint8_t {cname}_mask[{meta_size}];")
            h_lines.append(f"extern const uint16_t {cname}_slots[{meta_size}];")

        words = packed_data[lname]
        c_lines.append(f"const uint16_t {cname}_weights[{n_words}] = {{{', '.join(hex(x) for x in words)}}};")
        c_lines.append(f"const int32_t {cname}_bias[{layer['P']}] = {{{', '.join(str(int(x)) for x in layer['bias'])}}};")
        scale_list = layer.get("scale", [1.0] * layer["P"])
        
        c_lines.append(f"const float {cname}_scale[{len(scale_list)}] = {{")
        row = []
        for i, val in enumerate(scale_list):
            row.append(f"{float(val):.6f}f")
            if len(row) == 8 or i == len(scale_list) - 1:
                c_lines.append("    " + ", ".join(row) + ("," if i < len(scale_list) - 1 else ""))
                row = []
        c_lines.append("};")

        if max_d > 1:
            pos_array, mask_array, slots_array = [], [], []
            for word in result.words:
                for k in range(max_d):
                    if k < len(word.fields):
                        pos_array.append(word.fields[k].bit_offset)
                        mask_array.append((1 << word.fields[k].weight_bits) - 1)
                        slots_array.append(word.fields[k].slot_index)
                    else:
                        pos_array.append(0)
                        mask_array.append(0)
                        slots_array.append(0)
            c_lines.append(f"const uint8_t {cname}_pos[{meta_size}] = {{{', '.join(str(x) for x in pos_array)}}};")
            c_lines.append(f"const uint8_t {cname}_mask[{meta_size}] = {{{', '.join(hex(x) for x in mask_array)}}};")
            c_lines.append(f"const uint16_t {cname}_slots[{meta_size}] = {{{', '.join(str(x) for x in slots_array)}}};")

    h_lines.append("#endif")
    with open(header_path, "w") as f: f.write("\n".join(h_lines))
    with open(source_path, "w") as f: f.write("\n".join(c_lines))

def main():
    model = load_model('swin', checkpoint_path=os.path.join(PROJECT_ROOT, 'models', 'swin_model_cifar.pth'), num_classes=100)
    with open(os.path.join(PROJECT_ROOT, "quantization_framework_ViT", "configs", "swin_mqf_granular_config.json")) as f:
        quant_json = json.load(f)
    
    layers = extract_swin_layers(model, quant_json)
    
    packing_results, packed_data = {}, {}
    for layer in layers:
        lname = layer["name"]
        channels = list(zip(layer["w_bits"], layer["a_bits"]))
        if layer["K"] > MAX_K_FOR_SAFE_FFD or all(w == 8 for w, a in channels):
            from packing import pack_baseline
            result = pack_baseline(channels, r=REGISTER_WIDTH, layer_name=lname)
        else:
            result = pack_safe_ffd(channels, r=REGISTER_WIDTH, layer_name=lname)
        
        packing_results[lname] = result
        packed_data[lname] = build_packed_words(layer, result)
        print(f"Packed {lname} -> {result.n_words} words per filter")

    out_dir = os.path.join(THIS_DIR, "..", "generated")
    os.makedirs(out_dir, exist_ok=True)
    write_files(layers, packing_results, packed_data, out_dir)
    print("Done generating swin C-arrays!")

if __name__ == "__main__":
    main()
