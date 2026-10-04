import torch
from quantization_framework.models.mobilenet import mobilenet_v2
import __main__
__main__.MobileNetV2 = mobilenet_v2

m = mobilenet_v2(num_classes=10)

def safe_name(name):
    return name.replace(".", "_")

h = 32
w = 32

print("/* Auto-generated MobileNet Pipeline */")
print("int8_t *in_fm = fm_A;")
print("int8_t *out_fm = fm_B;")
print("int8_t *tmp;")

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
        
        func = "run_depthwise_conv2d_8bit" if is_dw else "run_conv2d_8bit"
        
        print(f"// {name} : {h}x{w}x{c_in} -> {out_h}x{out_w}x{c_out} (s={s}, p={p})")
        print(f"{func}(in_fm, out_fm, {safe_name(name)}_weights,")
        print(f"    {h}, {w}, {c_in},")
        if not is_dw:
            print(f"    {out_h}, {out_w}, {c_out},")
        else:
            print(f"    {out_h}, {out_w},")
        print(f"    {k}, {k}, {p}, {s},")
        print(f"    {safe_name(name)}_scale, {safe_name(name)}_bias);")
        print(f"apply_relu(out_fm, {out_h * out_w * c_out});")
        print("tmp = in_fm; in_fm = out_fm; out_fm = tmp;")
        print("")
        
        h = out_h
        w = out_w
        
    elif isinstance(module, torch.nn.Linear):
        c_in = module.in_features
        c_out = module.out_features
        print(f"// {name}")
        print(f"global_average_pool_2d(in_fm, out_fm, {h}, {w}, {c_in});")
        print("tmp = in_fm; in_fm = out_fm; out_fm = tmp;")
        print(f"run_linear_8bit(in_fm, out_fm, {safe_name(name)}_weights,")
        print(f"    {c_in}, {c_out},")
        print(f"    {safe_name(name)}_scale, {safe_name(name)}_bias);")
        print("in_fm = out_fm;")
        
