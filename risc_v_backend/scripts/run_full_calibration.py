import torch
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import types
from quantization_framework.models.mobilenet import MobileNetV2, mobilenet_v2
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.mobilenet"] = types.ModuleType("models.mobilenet")
sys.modules["models.mobilenet"].MobileNetV2 = MobileNetV2
sys.modules["models.mobilenet"].mobilenet_v2 = mobilenet_v2
import __main__
__main__.MobileNetV2 = MobileNetV2
__main__.mobilenet_v2 = mobilenet_v2

from quantization_framework.evaluation.pipeline import get_cifar10_dataloader

import json

print("Loading CIFAR-10 data...")
train_loader = get_cifar10_dataloader(train=True, batch_size=128, data_path=os.path.join(PROJECT_ROOT, "data"))

print("Loading uncalibrated QAT model...")
model = mobilenet_v2(num_classes=10)
ckpt = torch.load(os.path.join(PROJECT_ROOT, "models", "mobilenet_qat_weights.pth"), map_location="cpu")
model.load_state_dict(ckpt, strict=False)

print("Applying MQF Joint W=A Wrappers...")
with open(os.path.join(PROJECT_ROOT, "quantization_framework/configs/mobilenet_cifar10_channel_config_2_4_8_weight.json"), "r") as f:
    w_cfg = json.load(f)
with open(os.path.join(PROJECT_ROOT, "quantization_framework/configs/mobilenet_cifar10_channel_config_2_4_8_activation.json"), "r") as f:
    a_cfg = json.load(f)

# Note: validate_config.apply_mixed_precision creates the quantizers.
# For simplicity, we just manually inject the scales using our max hooking logic over 500 images
# because we know exactly what generate_packed_c.py expects.

input_scales = {}
output_scales = {}
hooks = []

import torch.nn as nn
def get_hook(name):
    def hook_fn(m, inp, out):
        x = inp[0]
        i_max = x.abs().max().item()
        o_max = out.abs().max().item()
        
        # Exponential moving average for max
        if name not in input_scales:
            input_scales[name] = i_max
            output_scales[name] = o_max
        else:
            input_scales[name] = 0.9 * input_scales[name] + 0.1 * i_max
            output_scales[name] = 0.9 * output_scales[name] + 0.1 * o_max
    return hook_fn

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hooks.append(module.register_forward_hook(get_hook(name)))

print("Calibrating on 5 batches...")
model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(train_loader):
        model(images)
        if i >= 4:
            break

for h in hooks:
    h.remove()

sd = model.state_dict()
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        sd[f"{name}.input_scale"] = torch.tensor(max(input_scales[name] / 127.0, 1e-9))
        sd[f"{name}.output_scale"] = torch.tensor(max(output_scales[name] / 127.0, 1e-9))
        w = module.weight.data
        w_max_per_ch = w.abs().amax(dim=tuple(range(1, w.dim())))
        w_scales = torch.clamp(w_max_per_ch / 127.0, min=1e-9)
        sd[f"{name}.weight._scale"] = w_scales

out_path = os.path.join(PROJECT_ROOT, "models", "mobilenet_fully_calibrated.pth")
torch.save(sd, out_path)
print(f"Calibration Complete! Model saved to {out_path}")

