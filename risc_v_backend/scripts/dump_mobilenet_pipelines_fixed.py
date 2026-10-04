import torch
from quantization_framework.models.mobilenet import mobilenet_v2
import __main__
__main__.MobileNetV2 = mobilenet_v2

m = mobilenet_v2(num_classes=10)

def safe_name(name):
    return name.replace(".", "_")

def generate_pipeline(packed=False):
    h = 32
    w = 32
    
    if packed:
        print("/* Auto-generated MobileNet Packed Pipeline */")
        print("int8_t *unpacked_fm = fm_B;")
        print("int8_t *packed_fm = fm_A;")
        print("int8_t *res_fm_ptr = res_fm;")
        print("uint32_t current_offsets[4096];")
        print("const uint8_t *prev_act_bits;")
        print("")
        print("// Initial state: unpacked_fm has the image.")
        print("for (int i=0; i<3072; i++) unpacked_fm[i] = cifar_image_0[i];")
        print("prev_act_bits = dummy_8_act_bits;")
        print("")
    else:
        print("/* Auto-generated MobileNet Baseline Pipeline */")
        print("int8_t *in_fm = fm_A;")
        print("int8_t *out_fm = fm_B;")
        print("int8_t *res_fm_ptr = res_fm;")
        print("int8_t *tmp;")
        print("")
        print("// Initial state: in_fm has the image.")
        print("for (int i=0; i<3072; i++) in_fm[i] = cifar_image_0[i];")
        print("")

    # Loop over all modules
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
            n = safe_name(name)
            
            # Check if this is the start of an InvertedResidual block
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'features' and parts[2] == 'conv':
                block_idx = int(parts[1])
                # The first conv in the block is either conv.0.0 (pw) or conv.0.0 (dw for expand_ratio=1)
                # But it's always the first module if it ends with .0.0 and there's no earlier one in this block.
                # Let's just hardcode: the first layer in `conv` is `conv[0]` which contains `Conv2d` as `0`.
                if name.endswith('.conv.0.0'):
                    block = m.features[block_idx]
                    if block.use_res_connect:
                        print(f"// --- Start of Residual Block {block_idx} ---")
                        if packed:
                            print(f"for (int i=0; i<{c_in * h * w}; i++) res_fm_ptr[i] = unpacked_fm[i];")
                        else:
                            print(f"for (int i=0; i<{c_in * h * w}; i++) res_fm_ptr[i] = in_fm[i];")
            
            print(f"// {name} : {h}x{w}x{c_in} -> {out_h}x{out_w}x{c_out} (s={s}, p={p})")
            
            if packed:
                func = "run_depthwise_conv2d_general" if is_dw else "run_conv2d_general"
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
            else:
                func = "run_depthwise_conv2d_8bit" if is_dw else "run_conv2d_8bit"
                print(f"{func}(in_fm, out_fm, {n}_weights,")
                print(f"    {h}, {w}, {c_in},")
                if not is_dw:
                    print(f"    {out_h}, {out_w}, {c_out},")
                else:
                    print(f"    {out_h}, {out_w},")
                print(f"    {k}, {k}, {p}, {s},")
                print(f"    {n}_scale, {n}_bias);")
            
            # ReLU is applied ONLY if name ends with .0 (meaning it is part of ConvBNReLU)
            if name.endswith('.0'):
                if packed:
                    print(f"apply_relu(unpacked_fm, {out_h * out_w * c_out});")
                else:
                    print(f"apply_relu(out_fm, {out_h * out_w * c_out});")
            
            if packed:
                print(f"prev_act_bits = {n}_out_act_bits;")
            else:
                print("tmp = in_fm; in_fm = out_fm; out_fm = tmp;")
            
            # Check if this is the END of an InvertedResidual block
            if len(parts) >= 3 and parts[0] == 'features' and parts[2] == 'conv':
                block_idx = int(parts[1])
                block = m.features[block_idx]
                is_last = False
                # The last Conv2d is the one before BatchNorm. In InvertedResidual, `block.conv` is a Sequential.
                # The second to last element is Conv2d.
                if block.conv[-2] == module:
                    is_last = True
                
                if is_last and block.use_res_connect:
                    print(f"// --- End of Residual Block {block_idx} ---")
                    if packed:
                        print(f"add_tensors(unpacked_fm, res_fm_ptr, {out_h * out_w * c_out});")
                    else:
                        print(f"add_tensors(in_fm, res_fm_ptr, {out_h * out_w * c_out});")
            
            print("")
            h = out_h
            w = out_w
            
        elif isinstance(module, torch.nn.Linear):
            c_in = module.in_features
            c_out = module.out_features
            n = safe_name(name)
            print(f"// {name}")
            if packed:
                print(f"global_average_pool_2d(unpacked_fm, packed_fm, {h}, {w}, {c_in});")
                print(f"// After pool, the data is in packed_fm but unpacked. So we just swap pointers.")
                print(f"int8_t *tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;")
                print(f"compute_packed_layout(prev_act_bits, current_offsets, {c_in}, 1);")
                print(f"pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, {c_in}, 1);")
                print(f"run_linear_general((const uint8_t*)packed_fm, unpacked_fm, {n}_weights,")
                print(f"    {n}_pos, {n}_mask, {n}_slots,")
                print(f"    prev_act_bits, current_offsets,")
                print(f"    {c_in}, {c_out},")
                print(f"    {n}_scale, {n}_bias,")
                print(f"    {n.upper()}_MAX_D, {n.upper()}_WORDS_PER_FILTER);")
            else:
                print(f"global_average_pool_2d(in_fm, out_fm, {h}, {w}, {c_in});")
                print("tmp = in_fm; in_fm = out_fm; out_fm = tmp;")
                print(f"run_linear_8bit(in_fm, out_fm, {n}_weights,")
                print(f"    {c_in}, {c_out},")
                print(f"    {n}_scale, {n}_bias);")
                print("in_fm = out_fm;")

if __name__ == "__main__":
    import sys
    packed = len(sys.argv) > 1 and sys.argv[1] == "--packed"
    generate_pipeline(packed)
