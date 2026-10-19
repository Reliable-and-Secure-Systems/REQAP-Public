# Pretrained Model Checkpoints

This directory holds the baseline floating-point (FP32) and legacy 8-bit checkpoints used by the REQAP quantization framework and firmware generators.

## Download Pretrained Weights

Please download the checkpoints from Google Drive and place them directly in this `models/` directory:

- 🔗 **Full Checkpoint Repository (FP32 Baselines & Joint Quantized):**  
  [Download All Checkpoints](https://drive.google.com/drive/folders/1epVQxwh2Qzx3l2wrKmSo5Q0LZj1-ej2L?usp=sharing)
- 🔗 **Legacy CNN Checkpoints (AlexNet, VGG-11, ResNet-18):**  
  [Download CNN Weights](https://drive.google.com/drive/folders/1R_EsvjougYKuMO2S44ArEI4xgYWuJ-bD?usp=sharing)
- 🔗 **Vision Transformer Checkpoints (Swin-T):**  
  [Download Transformer Weights](https://drive.google.com/drive/folders/1PTYVNfisoNvtrl_z6a_MxWqu5_-e01xl)

### Expected Checkpoint Files

| Model | Dataset | Expected Checkpoint Filename |
|---|---|---|
| **VGG-11** | CIFAR-10 | `vgg11_cifar10_fp32.pth` |
| **VGG-16** | CIFAR-10 | `vgg16_cifar10_fp32.pth` |
| **ResNet-18** | CIFAR-10 | `resnet18_cifar10_fp32.pth` |
| **AlexNet** | Fashion-MNIST | `alexnet_fashionmnist_fp32.pth` |
| **MobileNetV2** | CIFAR-10 | `mobilenet_cifar10_fp32.pth` |
| **Swin-T** | CIFAR-100 | `swin_cifar100_fp32.pth` / `best_swin_model_cifar_changed.pth` |
