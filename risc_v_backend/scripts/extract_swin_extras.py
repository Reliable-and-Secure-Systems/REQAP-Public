import os
import sys
import torch
import torch.nn as nn
from quantization_framework.models.swin import swin_tiny_patch4_window7_224

def safe_name(name):
    return name.replace(".", "_")

def extract_swin_extras():
    model = swin_tiny_patch4_window7_224(num_classes=10)
    # The weights might be random since we didn't load a checkpoint, but this is just for compiling the baseline!
    # If the user wants real weights, we load the checkpoint.
    checkpoint_path = os.path.join("models", "swin_model_cifar.pth")
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    h_file = open("risc_v_backend/generated/swin_extras.h", "w")
    c_file = open("risc_v_backend/generated/swin_extras.cc", "w")
    
    h_file.write("#ifndef SWIN_EXTRAS_H\n#define SWIN_EXTRAS_H\n#include <stdint.h>\n\n")
    c_file.write("#include \"swin_extras.h\"\n\n")

    # Extract LayerNorms
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            n = safe_name(name)
            dim = module.weight.shape[0]
            
            h_file.write(f"extern const int32_t {n}_weight[{dim}];\n")
            h_file.write(f"extern const int32_t {n}_bias[{dim}];\n")
            
            c_file.write(f"const int32_t {n}_weight[{dim}] = {{")
            # Quantize LN weight/bias to int32 (scale by 256 for 8-bit fractional precision)
            w = (module.weight.detach().numpy() * 256.0).astype(int)
            c_file.write(", ".join(map(str, w)))
            c_file.write("};\n")
            
            c_file.write(f"const int32_t {n}_bias[{dim}] = {{\n")
            b = (module.bias.detach().numpy() * 256.0).astype(int)
            c_file.write(", ".join(map(str, b)))
            c_file.write("};\n\n")

        elif isinstance(module, nn.Conv2d):
            n = safe_name(name)
            out_c = module.out_channels
            in_c = module.in_channels
            kh = module.kernel_size[0]
            kw = module.kernel_size[1]
            total_w = out_c * in_c * kh * kw
            
            h_file.write(f"extern const uint8_t {n}_weights[{total_w}];\n")
            h_file.write(f"extern const float {n}_scale[{out_c}];\n")
            h_file.write(f"extern const int32_t {n}_bias[{out_c}];\n")
            
            c_file.write(f"const uint8_t {n}_weights[{total_w}] = {{\n")
            # Fake 8-bit quantization for baseline dummy
            w = module.weight.detach().numpy().flatten()
            w_q = ((w - w.min()) / (w.max() - w.min() + 1e-9) * 255.0).astype(int)
            c_file.write(", ".join(map(str, w_q)))
            c_file.write("};\n")
            
            c_file.write(f"const float {n}_scale[{out_c}] = {{\n")
            s = [0.01] * out_c
            c_file.write(", ".join(map(str, s)))
            c_file.write("};\n")
            
            c_file.write(f"const int32_t {n}_bias[{out_c}] = {{\n")
            if module.bias is not None:
                b = (module.bias.detach().numpy() * 256.0).astype(int)
            else:
                b = [0] * out_c
            c_file.write(", ".join(map(str, b)))
            c_file.write("};\n\n")

    # Extract relative position biases
    for name, param in model.named_parameters():
        if "relative_position_bias_table" in name:
            n = safe_name(name)
            size = param.numel()
            h_file.write(f"extern const int32_t {n}[{size}];\n")
            
            c_file.write(f"const int32_t {n}[{size}] = {{")
            # Quantize rel pos bias to int32 (scale by 256)
            v = (param.detach().numpy().flatten() * 256.0).astype(int)
            c_file.write(", ".join(map(str, v)))
            c_file.write("};\n\n")

    h_file.write("#endif\n")
    h_file.close()
    c_file.close()
    print("Exported swin_extras.h and swin_extras.cc")

if __name__ == "__main__":
    extract_swin_extras()
