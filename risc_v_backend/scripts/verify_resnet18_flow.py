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
from quantization_framework.models.resnet import ResNet, ResNet18, BasicBlock
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.resnet"] = types.ModuleType("models.resnet")
sys.modules["models.resnet"].ResNet = ResNet
sys.modules["models.resnet"].ResNet18 = ResNet18
sys.modules["models.resnet"].BasicBlock = BasicBlock
import __main__
__main__.ResNet = ResNet
__main__.ResNet18 = ResNet18
__main__.BasicBlock = BasicBlock

# Load model
pth_path = os.path.join(PROJECT_ROOT, "models", "resnet18.pt")
checkpoint = torch.load(pth_path, map_location="cpu")
if isinstance(checkpoint, nn.Module):
    model = checkpoint
else:
    model = ResNet18(num_classes=10)
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
def print_stat(name, tensor):
    print(f"  {name:<25}: shape={str(list(tensor.shape)):<18} min={tensor.min().item():>8.4f} max={tensor.max().item():>8.4f} mean={tensor.abs().mean().item():>8.4f}")

x = model.conv1(x)
x = model.bn1(x)
x = model.relu(x)
print_stat("conv1", x)
x = model.maxpool(x)

for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
    for block_idx, block in enumerate(layer):
        x = block(x)
        print_stat(f"layer{layer_idx}[{block_idx}]", x)

x = model.avgpool(x)
x = torch.flatten(x, 1)
x = model.fc(x)

print("\n=== Final Logits ===")
print_stat("fc (Logits)", x)
for i in range(10):
    print(f"Class {i}: {x[0, i].item():.4f}")
print("Predicted Class:", x.argmax().item())
