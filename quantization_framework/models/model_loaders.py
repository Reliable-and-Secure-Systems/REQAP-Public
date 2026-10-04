import os
import torch

# Default to looking for 'models' relative to the project root, or use env var
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(PROJECT_ROOT, '../models'))

def get_model_size_info(model, checkpoint_path=None):
    """
    Calculate model size in MB.
    Returns dict with size information.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate memory size (assuming FP32)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Get file size if checkpoint provided
    file_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    return {
        'parameters': total_params,
        'size_mb': size_mb,
        'file_size_mb': file_size_mb
    }


def print_model_size(model, checkpoint_path=None, label="Model"):
    """Print model size information in a clean format."""
    info = get_model_size_info(model, checkpoint_path)
    
    print(f"{'='*60}")
    print(f"{label} Size:")
    print(f"  Parameters: {info['parameters']:,} | Memory: {info['size_mb']:.2f} MB", end="")
    if info['file_size_mb']:
        print(f" | File: {info['file_size_mb']:.2f} MB")
    else:
        print()
    print(f"{'='*60}")

def load_model(model_name, checkpoint_path=None, num_classes=10):
    """
    Load a model by name, optionally loading weights from a checkpoint.
    """
    model = None
    if model_name == 'vgg11_bn':
        from .vgg import vgg11_bn
        model = vgg11_bn(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'vgg11_cifar10_fp32.pth')
    elif model_name == 'levit':
        from .levit import levit_cifar
        model = levit_cifar(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'best3_levit_model_cifar10.pth')
    elif model_name == 'swin':
        from .swin import swin_tiny_patch4_window7_224
        # Warning: Swin architectures vary greatly. This is a best-effort load.
        model = swin_tiny_patch4_window7_224(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'swin_cifar100_fp32.pth')
    elif model_name == 'resnet':
        from .resnet import ResNet18
        cifar_flag = (num_classes in [10, 100])
        model = ResNet18(num_classes=num_classes, cifar=cifar_flag)
        default_ckpt = os.path.join(MODELS_DIR, 'resnet18_cifar10_fp32.pth')
    elif model_name == 'alexnet':
        from .alexnet import AlexNet
        model = AlexNet(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'alexnet_fashionmnist_fp32.pth')
    elif model_name == 'mobilenet':
        from .mobilenet import MobileNetV2
        model = MobileNetV2(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'mobilenet_cifar10_fp32.pth')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load weights if path provided or default exists
    ckpt_to_load = checkpoint_path if checkpoint_path else default_ckpt
    
    if os.path.exists(ckpt_to_load):
        print(f"Loading checkpoint from {ckpt_to_load}")
        try:
            state_dict = torch.load(ckpt_to_load, map_location='cpu')
            # Handle possible key mismatches (e.g. 'module.' prefix)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
                
            # Remove module. prefix if present (DataParallel)
            # Also remove model. prefix (common in some training wrappers)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                if name.startswith('model.'):
                    name = name[6:]
                new_state_dict[name] = v
                
            # Strict=False to allow for minor architecture mismatches during research dev
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        except Exception as e:
            # Previously this printed a message and silently continued with
            # a RANDOMLY-INITIALIZED model — the checkpoint existing means
            # the caller expected real weights, so a load failure here
            # (e.g. classifier shape mismatch from a wrong num_classes /
            # wrong-dataset pairing) must never be tolerated silently. This
            # exact failure mode produced the two unreproduced CIFAR-100
            # rows in the paper (LeViT/Swin ran against CIFAR-10 with a
            # partially-loaded 100-class head) — see REMEDIATION_PLAN.md
            # Phase A4 / audit Finding 5.
            raise RuntimeError(
                f"Failed to load checkpoint {ckpt_to_load} for model_name={model_name!r}, "
                f"num_classes={num_classes}: {e}"
            ) from e
    else:
        print(f"Checkpoint not found at {ckpt_to_load}, returning random init model.")
        
    print_model_size(model, ckpt_to_load, label=f"{model_name}")
    
    return model
