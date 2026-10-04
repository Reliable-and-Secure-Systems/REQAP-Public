import torch
from quantization_framework.models.mobilenet import mobilenet_v2
import __main__
__main__.MobileNetV2 = mobilenet_v2

m = mobilenet_v2(num_classes=10)

def safe_name(name):
    return name.replace(".", "_")

h = 32
w = 32

print("/* Auto-generated MobileNet Packed Pipeline */")
print("int8_t *unpacked_fm = fm_B;")
print("int8_t *packed_fm = fm_A;")
print("uint32_t current_offsets[4096];")
print("const uint8_t *prev_act_bits;")
print("")
print("// Initial state: unpacked_fm has the image.")
print("for (int i=0; i<3072; i++) unpacked_fm[i] = cifar_image_0[i];")
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
        
        is_dw = (g == c_in and c_in == c_out)
        
        func = "run_depthwise_conv2d_general" if is_dw else "run_conv2d_general"
        n = safe_name(name)
        
        print(f"// {name} : {h}x{w}x{c_in} -> {out_h}x{out_w}x{c_out} (s={s}, p={p})")
        print(f"compute_packed_layout(prev_act_bits, current_offsets, {c_in}, {h} * {w});")
        print(f"pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, {c_in}, {h} * {w});")
        
        print(f"{func}((const uint8_t*)packed_fm, unpacked_fm, {n}_weights,")
        print(f"    {n}_pos, {n}_mask, {n}_slots,")
        print(f"    prev_act_bits, current_offsets,")
        print(f"    {h}, {w}, {c_in},")
        if not is_dw:
            print(f"    {out_h}, {out_w}, {c_out},")
        else:
            print(f"    {out_h}, {out_w},")
        print(f"    {k}, {k}, {p}, {s},")
        print(f"    {n}_scale, {n}_bias,")
        print(f"    {n.upper()}_MAX_D, {n.upper()}_WORDS_PER_FILTER);")
        
        print(f"apply_relu(unpacked_fm, {out_h * out_w * c_out});")
        print(f"prev_act_bits = {n}_out_act_bits;")
        print("")
        
        h = out_h
        w = out_w
        
    elif isinstance(module, torch.nn.Linear):
        c_in = module.in_features
        c_out = module.out_features
        n = safe_name(name)
        print(f"// {name}")
        print(f"global_average_pool_2d(unpacked_fm, packed_fm, {h}, {w}, {c_in});")
        print(f"// After pool, the data is in packed_fm but unpacked. So we just swap pointers.")
        print(f"int8_t *tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;")
        print(f"compute_packed_layout(prev_act_bits, current_offsets, {c_in}, 1);")
        print(f"pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, {c_in}, 1);")
        
        print(f"run_linear_general((const uint8_t*)packed_fm, unpacked_fm, {n}_weights,")
        print(f"    {n}_pos, {n}_mask, {n}_slots,")
        print(f"    prev_act_bits, current_offsets,")
        print(f"    {c_in}, {c_out},")
        print(f"    {n}_scale, {n}_bias,")
        print(f"    {n.upper()}_MAX_D, {n.upper()}_WORDS_PER_FILTER);")
        
