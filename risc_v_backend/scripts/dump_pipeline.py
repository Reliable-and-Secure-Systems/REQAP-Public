import torch
import sys
from quantization_framework.models.vgg import vgg11_bn
from quantization_framework.models.resnet import resnet18
from quantization_framework.models.alexnet import alexnet

import __main__
__main__.VGG = vgg11_bn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Model name (e.g. vgg11)")
parser.add_argument("--baseline", action="store_true", help="Dump baseline unpacked pipeline")
args = parser.parse_args()

model_name = args.model_name
is_baseline = args.baseline

if model_name == "vgg11":
    m = vgg11_bn(num_classes=10)
elif model_name == "resnet18":
    m = resnet18(num_classes=10)
elif model_name == "alexnet":
    m = alexnet(num_classes=10)

def safe_name(name):
    return name.replace(".", "_")

if model_name == "alexnet":
    h = 227
    w = 227
else:
    h = 32
    w = 32

print(f"/* Auto-generated {model_name} Pipeline (Baseline={is_baseline}) */")
print("int8_t *unpacked_fm = fm_B;")
print("int8_t *packed_fm = fm_A;")
print("int8_t *tmp;")
if not is_baseline:
    print("uint32_t current_offsets[4096];")
    print("uint8_t dummy_8_act_bits[4096];")
    print("for (int _i = 0; _i < 4096; _i++) dummy_8_act_bits[_i] = 8;")
    print("const uint8_t *prev_act_bits;")
print("")
print("// Initial state: unpacked_fm has the image.")
c_in_init = 1 if model_name == "alexnet" else 3
print(f"for (int i=0; i<{h * w * c_in_init}; i++) unpacked_fm[i] = cifar_image_0[i % 3072];")
if not is_baseline:
    print("prev_act_bits = dummy_8_act_bits;")
print("")

for name, module in m.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        c_in = module.in_channels
        c_out = module.out_channels
        k = module.kernel_size[0]
        s = module.stride[0]
        p = module.padding[0]
        g = module.groups
        
        out_h = (h + 2*p - k) // s + 1
        out_w = (w + 2*p - k) // s + 1
        
        n = safe_name(name)
        
        print(f"// {name} : {h}x{w}x{c_in} -> {out_h}x{out_w}x{c_out} (s={s}, p={p})")
        if not is_baseline:
            func = "run_conv2d_general"
            print(f"compute_packed_layout(prev_act_bits, current_offsets, {c_in}, {h} * {w});")
            print(f"pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, {c_in}, {h} * {w});")
            print(f"{func}((const uint8_t*)packed_fm, unpacked_fm, {n}_weights,")
            print(f"    {n}_pos, {n}_mask, {n}_slots,")
            print(f"    prev_act_bits, current_offsets,")
            print(f"    {h}, {w}, {c_in},")
            print(f"    {out_h}, {out_w}, {c_out},")
            print(f"    {k}, {k}, {p}, {s},")
            print(f"    {n}_scale, {n}_bias,")
            print(f"    {n.upper()}_MAX_D, {n.upper()}_WORDS_PER_FILTER);")
        else:
            func = "run_conv2d_8bit"
            print(f"{func}(unpacked_fm, packed_fm, {n}_weights,")
            print(f"    {h}, {w}, {c_in},")
            print(f"    {out_h}, {out_w}, {c_out},")
            print(f"    {k}, {k}, {p}, {s},")
            print(f"    {n}_scale, {n}_bias);")
            print(f"tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;")
            
        if "relu" in str(module) or True:  # Simplify for now, just apply relu after conv unless specified
            if not ("downsample" in name): # ResNet shortcut doesn't have relu directly here
                print(f"apply_relu(unpacked_fm, {out_h * out_w * c_out});")
        
        if not is_baseline:
            print(f"prev_act_bits = {n}_out_act_bits;")
        print("")
        
        h = out_h
        w = out_w
        
    elif isinstance(module, torch.nn.MaxPool2d):
        k = module.kernel_size
        s = module.stride
        p = module.padding
        out_h = (h + 2*p - k) // s + 1
        out_w = (w + 2*p - k) // s + 1
        print(f"// {name} (MaxPool)")
        print(f"max_pool_2d(unpacked_fm, packed_fm, {h}, {w}, {c_out});")
        print(f"tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;")
        h = out_h
        w = out_w

    elif isinstance(module, torch.nn.Linear):
        c_in = module.in_features
        c_out = module.out_features
        n = safe_name(name)
        print(f"// {name}")
        if model_name != "alexnet": # alexnet has avgpool before linear? No
            pass 
        
        if not is_baseline:
            print(f"compute_packed_layout(prev_act_bits, current_offsets, {c_in}, 1);")
            print(f"pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, {c_in}, 1);")
            
            print(f"run_linear_general((const uint8_t*)packed_fm, unpacked_fm, {n}_weights,")
            print(f"    {n}_pos, {n}_mask, {n}_slots,")
            print(f"    prev_act_bits, current_offsets,")
            print(f"    {c_in}, {c_out},")
            print(f"    {n}_scale, {n}_bias,")
            print(f"    {n.upper()}_MAX_D, {n.upper()}_WORDS_PER_FILTER);")
        else:
            print(f"run_linear_8bit(unpacked_fm, packed_fm, {n}_weights,")
            print(f"    {c_in}, {c_out},")
            print(f"    {n}_scale, {n}_bias);")
            print(f"tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;")
            
        if c_out != 10: # Not the final classifier
            print(f"apply_relu(unpacked_fm, {c_out});")
            if not is_baseline:
                print(f"prev_act_bits = {n}_out_act_bits;")
        print("")
