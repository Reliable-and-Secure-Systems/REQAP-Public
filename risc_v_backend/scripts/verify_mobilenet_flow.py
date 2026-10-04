import os
import sys
import torch
import torch.nn as nn
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Inject classes so torch.load works
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

# Load image from header
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

image_tensor = torch.tensor(image_vals, dtype=torch.float32).view(1, 3, 32, 32)
# The C implementation casts int8 values natively, we must respect the int8 wrapping if the values in header are unsigned!
# Wait, cifar_image_0 is defined as uint8_t, but is it [-128, 127] in logic? 
# Usually C will cast `uint8_t` (0..255) into `int8_t` (-128..127) using two's complement.
# Let's see the cast logic:
casted_vals = []
for v in image_vals:
    if v > 127:
        casted_vals.append(v - 256)
    else:
        casted_vals.append(v)
image_tensor = torch.tensor(casted_vals, dtype=torch.float32).view(1, 3, 32, 32)
print("Input image range: min=", image_tensor.min().item(), "max=", image_tensor.max().item(), "mean=", image_tensor.abs().mean().item())

# Trace layer-by-layer forward pass
print("\n=== Trace PyTorch Forward Pass ===")
x = image_tensor
layers = list(model.features)
for idx, layer in enumerate(layers):
    x = layer(x)
    name = f"features.{idx}"
    print(f"  {name:<25}: shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")

# Average Pool
x = x.mean([2, 3])
print(f"  features.AvgPool         : shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")

# Flatten
x = torch.flatten(x, 1)

# Classifier
classifiers = list(model.classifier)
for idx, layer in enumerate(classifiers):
    x = layer(x)
    name = f"classifier.{idx}"
    print(f"  {name:<25}: shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")

print("\nFinal Logits:", x.detach().numpy())
print("Predicted Class:", x.argmax(dim=1).item())

