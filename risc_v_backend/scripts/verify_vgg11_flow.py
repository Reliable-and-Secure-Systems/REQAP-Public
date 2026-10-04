import os
import sys
import torch
import torch.nn as nn
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Inject classes so torch.load works
import types
from quantization_framework.models.vgg import VGG, vgg11_bn
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.vgg"] = types.ModuleType("models.vgg")
sys.modules["models.vgg"].VGG = VGG
sys.modules["models.vgg"].vgg11_bn = vgg11_bn
import __main__
__main__.VGG = VGG
__main__.vgg11_bn = vgg11_bn

# Load model
pth_path = os.path.join(PROJECT_ROOT, "models", "qvgg-8bit.pth")
checkpoint = torch.load(pth_path, map_location="cpu")
if isinstance(checkpoint, nn.Module):
    model = checkpoint
else:
    model = vgg11_bn(num_classes=10)
    model.load_state_dict(checkpoint, strict=False)
model.eval()

# Load image from header
image_vals = []
with open(os.path.join(THIS_DIR, "generated", "cifar_test_image.h"), "r") as f:
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
# Keep it in [-128, 127] for direct matching to CIFAR C array
print("Input image range: min=", image_tensor.min().item(), "max=", image_tensor.max().item(), "mean=", image_tensor.abs().mean().item())

# Trace layer-by-layer forward pass
print("\n=== Trace PyTorch Forward Pass ===")
x = image_tensor
layers = list(model.features)
for idx, layer in enumerate(layers):
    x = layer(x)
    name = f"features.{idx}"
    print(f"  {name:<25}: shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")

# AdaptiveAvgPool2d
pool = nn.AdaptiveAvgPool2d((1, 1))
x = pool(x)
print(f"  features.MaxPool (1x1)   : shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")

# Flatten
x = torch.flatten(x, 1)

# Classifier
classifiers = list(model.classifier)
for idx, layer in enumerate(classifiers):
    x = layer(x)
    name = f"classifier.{idx}"
    print(f"  {name:<25}: shape={str(list(x.shape)):<18} min={x.min().item():>8.4f} max={x.max().item():>8.4f} mean={x.abs().mean().item():>8.4f}")
