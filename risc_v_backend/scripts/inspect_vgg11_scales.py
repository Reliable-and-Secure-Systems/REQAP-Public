import os
import sys
import torch
import types

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantization_framework.models.vgg import VGG, vgg11_bn
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.vgg"] = types.ModuleType("models.vgg")
sys.modules["models.vgg"].VGG = VGG
sys.modules["models.vgg"].vgg11_bn = vgg11_bn
import __main__
__main__.VGG = VGG
__main__.vgg11_bn = vgg11_bn

pth_path = os.path.join(PROJECT_ROOT, "models", "qvgg-8bit.pth")
checkpoint = torch.load(pth_path, map_location="cpu")

print("=== Checkpoint Type ===")
print(type(checkpoint))

if isinstance(checkpoint, torch.nn.Module):
    model = checkpoint
    state_dict = model.state_dict()
else:
    state_dict = checkpoint

print("\n=== All State Dict Keys ===")
for k in sorted(state_dict.keys()):
    if "scale" in k or "bias" in k:
        print(f"  {k}: shape={list(state_dict[k].shape) if hasattr(state_dict[k], 'shape') else 'scalar'} val={state_dict[k]}")
