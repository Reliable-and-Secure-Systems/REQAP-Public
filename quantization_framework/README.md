# MQF Quantization Framework

This module is the first stage in the pipeline. It parses standard PyTorch CNNs (VGG-11, ResNet-18, AlexNet) and executes Post-Training Quantization (PTQ) to determine the optimal low-bit representations for the weights.

## Execution Flow
Rather than assigning a uniform bit-width to the entire model, this framework analyzes the variance of individual blocks of weights (granules) and generates heterogeneous JSON configuration maps (assigning 2-bit, 4-bit, or 8-bit widths depending on the sensitivity of that specific granule).

## How to Run

Before running the experiments, make sure you are in the `quantization_framework/` directory and have installed the root `requirements.txt`.

### 1. Generating Configuration Maps
To run the full mixed-precision study and export the optimal JSON configuration for a specific model, run:

```bash
# Example for VGG-11
python experiments/run_full_mixed_precision_study.py --model vgg11

# Example for ResNet-18
python experiments/run_full_mixed_precision_study.py --model resnet18
```

The generated configuration maps will be saved as JSON files in the `configs/` directory. These maps dictate exactly how the hardware simulator should pack the weights.

### 2. Validating a Configuration
If you want to manually test the classification accuracy of a specific JSON configuration map to ensure the accuracy degradation is within acceptable limits, run:

```bash
python experiments/validate_config.py --model vgg11 --config configs/vgg11_bn_config_2_4_8.json
```
