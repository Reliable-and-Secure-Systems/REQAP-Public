import torch
import torch.nn as nn
import os
import sys

# Inject classes
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import types
from quantization_framework.models.mobilenet import MobileNetV2, mobilenet_v2
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.mobilenet"] = types.ModuleType("models.mobilenet")
sys.modules["models.mobilenet"].MobileNetV2 = MobileNetV2
sys.modules["models.mobilenet"].mobilenet_v2 = mobilenet_v2
import __main__
__main__.MobileNetV2 = MobileNetV2
__main__.mobilenet_v2 = mobilenet_v2

# Load model
pth_path = os.path.join(PROJECT_ROOT, "models", "mobilenet_qat_weights.pth")
checkpoint = torch.load(pth_path, map_location="cpu")
model = mobilenet_v2(num_classes=10)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Load CIFAR test image
image_vals = []
with open(os.path.join(PROJECT_ROOT, "risc_v_backend", "firmware", "generated", "cifar_test_image.h"), "r") as f:
    for line in f:
        if "cifar_image_0" in line or "const" in line or "uint8_t" in line or "};" in line or "/*" in line or "#include" in line or line.strip() == "":
            continue
        parts = line.replace(",", " ").strip().split()
        for p in parts:
            if p:
                try:
                    image_vals.append(int(p))
                except ValueError:
                    pass
casted_vals = []
for v in image_vals:
    if v > 127:
        casted_vals.append(v - 256)
    else:
        casted_vals.append(v)
image_tensor = torch.tensor(casted_vals, dtype=torch.float32).view(1, 3, 32, 32)

# Hook and Calibrate
input_scales = {}
output_scales = {}
hooks = []

def get_hook(name):
    def hook_fn(m, inp, out):
        x = inp[0]
        i_max = x.abs().max().item()
        o_max = out.abs().max().item()
        input_scales[name] = max(i_max / 127.0, 1e-9)
        output_scales[name] = max(o_max / 127.0, 1e-9)
    return hook_fn

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hooks.append(module.register_forward_hook(get_hook(name)))

with torch.no_grad():
    model(image_tensor)

for h in hooks:
    h.remove()

# Inject into State Dict
sd = model.state_dict()
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        sd[f"{name}.input_scale"] = torch.tensor(input_scales[name])
        sd[f"{name}.output_scale"] = torch.tensor(output_scales[name])
        w = module.weight.data
        w_max_per_ch = w.abs().amax(dim=tuple(range(1, w.dim())))
        w_scales = torch.clamp(w_max_per_ch / 127.0, min=1e-9)
        sd[f"{name}.weight._scale"] = w_scales

# Save
out_path = os.path.join(PROJECT_ROOT, "models", "mobilenet_calibrated_8bit.pth")
torch.save(sd, out_path)
print(f"Saved calibrated model to {out_path}")
