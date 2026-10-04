# REQAP (Reliable and Efficient Quantization-Aware Packing for Deep Neural Networks)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**A novel neural network compression and execution framework that combines granular sensitivity-driven quantization with Safe-FFD hardware register packing for ultra-efficient RISC-V edge inference.**

## Overview

This repository contains the complete end-to-end pipeline for the MQF (Multi-Quantization Framework), which introduces:

- **Filter-level sensitivity analysis** to assign optimal granular bit-widths (supports arbitrary 2-8 bit precisions)
- **Joint Weight and Activation (W=A) co-optimization** targeting exact hardware register sizes
- **Safe-FFD bin-packing algorithm** to maximize memory density in 16-bit hardware registers
- **Bare-metal RISC-V C/ASM deployment** utilizing SIMD Within A Register (SWAR) arithmetic

## CNN Architecture & Code Map

To understand the end-to-end flow from high-level PyTorch down to bare-metal C execution on the VC707 FPGA, please refer to the following code map:

1. **Unquantified FP32 Baseline (Python)**: [`quantization_framework/models/vgg.py`](file:///C:/Mubashir-BTU/Thesis/Codes/Danial/REQAP-DNN/quantization_framework/models/vgg.py) - This file contains the standard, pure 32-bit floating-point PyTorch implementation of the VGG-11 architecture.
2. **MQF Multi-precision Quantization (Python)**: [`quantization_framework/quantize_models.py`](file:///C:/Mubashir-BTU/Thesis/Codes/Danial/REQAP-DNN/quantization_framework/quantize_models.py) - This script applies the MQF algorithm to the baseline model, determining granular bit-widths and simulating the accuracy impact.
3. **Baseline 8-bit Execution (C)**: `run_full_vgg11_baseline_forward_pass()` in [`vc707_deployment/main_vc707_vgg11.c`](file:///C:/Mubashir-BTU/Thesis/Codes/Danial/REQAP-DNN/vc707_deployment/main_vc707_vgg11.c) - The standard 8-bit integer inference implementation acting as our control group.
4. **MQF Multi-precision Execution (C)**: `run_full_vgg11_packed_forward_pass()` in [`vc707_deployment/main_vc707_vgg11.c`](file:///C:/Mubashir-BTU/Thesis/Codes/Danial/REQAP-DNN/vc707_deployment/main_vc707_vgg11.c) - The optimized deployment that uses the custom SWAR MAC kernels ([`swar_mac.c`](file:///C:/Mubashir-BTU/Thesis/Codes/Danial/REQAP-DNN/risc_v_backend/firmware/swar_mac.c)) to multiply the packed multi-precision weights on the FPGA.

### Network Layer Structures
Our framework currently supports the heterogeneous quantization of the following three core visual architectures:

- **VGG-11**: `[Input: 3x32x32]` → `[Conv2D (64) + ReLU + MaxPool]` → `[Conv2D (128) + ReLU + MaxPool]` → `[2x Conv2D (256) + ReLU + MaxPool]` → `[2x Conv2D (512) + ReLU + MaxPool]` → `[2x Conv2D (512) + ReLU + MaxPool]` → `[FC (4096)]` → `[FC (4096)]` → `[FC (10)]`
- **ResNet-18**: `[Input: 3x32x32]` → `[Conv2D (64) + BatchNorm + ReLU]` → `[4x BasicBlock layers with Skip Connections (64 → 128 → 256 → 512)]` → `[AdaptiveAvgPool2D]` → `[FC (10)]`
- **AlexNet**: `[Input: 1x28x28]` → `[Conv2D (64) + ReLU + MaxPool]` → `[Conv2D (192) + ReLU + MaxPool]` → `[3x Conv2D (384/256/256) + ReLU + MaxPool]` → `[AdaptiveAvgPool2D]` → `[FC (4096)]` → `[FC (4096)]` → `[FC (10)]`

*(Note: ResNet-18 and AlexNet follow this exact same PyTorch-to-C pipeline structure in their respective files).*

### Key Results

By evaluating the MQF approach with a 5% accuracy drop constraint, we demonstrated substantial improvements in inference efficiency through our custom Safe-FFD packing logic targeting 16-bit registers.

| Architecture | Dataset | Baseline Accuracy | MQF Accuracy | Accuracy Drop | Register Fetch Reduction |
|-------------|---------|-------------------|--------------|---------------|--------------------------|
| **AlexNet** | Fashion-MNIST | 88.54% | 86.82% | -1.72% | **~51.2%** |
| **VGG-11** | CIFAR-10 | 92.11% | 89.45% | -2.66% | **~44.8%** |
| **ResNet-18**| CIFAR-10 | 94.62% | 90.81% | -3.81% | **~41.3%** |
| **MobileNetV2** | CIFAR-100 | TBD | TBD | TBD | **TBD** |
| **Swin-T‡** | CIFAR-100 | 89.13% | 88.81% | -0.32% | **~38.7%** |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Phase 1: PTQ Configuration](#phase-1-mqf-ptq-configuration)
  - [Phase 2: Hardware Simulation](#phase-2-systolic-hardware-simulation)
  - [Phase 3: RISC-V Deployment](#phase-3-risc-v-bare-metal-deployment)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Hardware Dependency (Chipyard)

To achieve true hardware performance verification, our bare-metal RISC-V C-code targets the physical VC-707 FPGA environment. 

The custom hardware board, including the SoC architecture, boot configs, and linker scripts, was synthesized and prepared using [Chipyard](https://github.com/ucb-bar/chipyard). Because the Chipyard infrastructure is massive, we have decoupled it from this framework repo. 

Instead, we use a custom fork of Chipyard hosted at:
**[RSS-Chipyard (VC707)](https://github.com/Reliable-and-Secure-Systems/Chipyard/tree/main/fpga/src/main/scala/vc707)**

**Reproducibility Note:**
*This* repository (REQAP-DNN) does not synthesize the hardware bitstream. Instead, it consumes the hardware configuration generated by our Chipyard clone. By linking against the `boot1.S` and `link_uart.ld` configurations provided by Chipyard, the `vc707_deployment/Makefile` strictly conforms to the precise memory addresses required to successfully boot and execute the MQF-packed models directly on the FPGA.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- RISC-V Cross-Compiler Toolchain (`riscv-none-elf-gcc`)
- QEMU Emulator (`qemu-system-misc`)
- 8GB RAM minimum

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Reliable-and-Secure-Systems/REQAP-Public.git
cd REQAP-Public

# Install Python requirements
pip install -r requirements.txt
```

### Install RISC-V Toolchain & QEMU

- **Windows:** Download pre-built binaries from [SiFive/xPack](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack) and [QEMU](https://qemu.weilnetz.de/w64/). Add their `bin` directories to your system `PATH`.
- **Linux:** 
```bash
sudo apt install gcc-riscv64-unknown-elf
sudo apt install qemu-system-misc
```

---

## Quick Start

> **⚠️ IMPORTANT: Pretrained Models Required**
>
> This repository requires pretrained `.pth` or `.pt` PyTorch model checkpoints to generate valid quantization configurations. Place your trained models in the `models/` directory.

### 1. Download Pretrained Models
Ensure your trained models are stored in the root `models/` folder. You can download our pretrained weights from the links below:

* **Latest CNN and Vision Transformer Models (FP32 Baselines & Joint W=A Quantized):**  
  [Download All Checkpoints](https://drive.google.com/drive/folders/1epVQxwh2Qzx3l2wrKmSo5Q0LZj1-ej2L?usp=sharing)

* **Legacy CNN Models (AlexNet, VGG11, ResNet18 - 8-bit):**  
  [Download CNN Checkpoints](https://drive.google.com/drive/folders/1R_EsvjougYKuMO2S44ArEI4xgYWuJ-bD?usp=sharing)
  
* **Legacy Vision Transformer Models (Swin-T):**  
  [Download Transformer Checkpoints](https://drive.google.com/drive/folders/1PTYVNfisoNvtrl_z6a_MxWqu5_-e01xl)
### 2. Generate MQF Configurations
```bash
cd quantization_framework
python experiments/run_full_mixed_precision_study.py --model vgg11
```

### 3. Compile and Run on RISC-V QEMU
```bash
cd risc_v_backend
python scripts/generate_packed_c.py --model VGG11
make -C firmware vgg11_packed
make -C firmware run_vgg11_packed

# Or for Vision Transformers (Swin-T):
python scripts/generate_swin_packed_c.py
make -C firmware swin_packed
make -C firmware run_swin_packed
```

---

## Dataset Setup

### CIFAR-10

Automatically downloaded by PyTorch `torchvision` on the first run of the quantization evaluation pipelines. No manual setup is required.

---

## Usage

The framework operates in three sequential stages. You must run them in order.

### Phase 1: MQF PTQ Configuration

The `quantization_framework` directory handles the Python-based Post-Training Quantization (PTQ) to generate custom MQF bit-width maps.

```bash
cd quantization_framework

# Run the greedy search and sensitivity analysis
python experiments/joint_search.py \
    --model vgg11 \
    --checkpoint ../models/vgg-8bit.pth \
    --dataset cifar10 \
    --target-drop 5.0
```

### Phase 2: Systolic Hardware Simulation

The `systolic_sim` directory contains the Python hardware simulator that executes the Safe-FFD bin-packing algorithm to extract theoretical memory fetch metrics.

```bash
cd systolic_sim

# Simulate hardware packing and memory reductions
python simulator.py
```

### Phase 3: RISC-V Bare-Metal Deployment

The `risc_v_backend` directory translates the PyTorch models and configuration JSONs into C-arrays, and executes them on RISC-V using SWAR logic.

```bash
cd risc_v_backend

# 1. Generate the C-arrays (Choose CNN or Transformer)
# By default, this uses the MQF JSON configuration for heterogeneous packing.
python scripts/generate_packed_c.py --model VGG11

# To generate a standard un-packed 8-bit model for baseline performance comparisons,
# use the --baseline flag (this ignores the JSON config and forces 8-bit uniformity):
# python scripts/generate_packed_c.py --model VGG11 --baseline

python scripts/generate_swin_packed_c.py

# 2. Compile the RISC-V firmware
make -C firmware vgg11_packed
make -C firmware swin_packed

# 3. Run the inference on QEMU
make -C firmware run_vgg11_packed
make -C firmware run_swin_packed
```

### Phase 4: Interactive VC707 Hardware-in-the-Loop Demonstration UI

The `VC707_Live_Demo_UI` directory contains a standalone interactive web dashboard (Streamlit) and real-time UART telemetry bridge for live FPGA presentations:

```bash
cd VC707_Live_Demo_UI/python_ui

# Install UI requirements
pip install -r requirements.txt

# Launch the live interactive dashboard
streamlit run app.py
# Or double-click run_demo.bat (Windows) / ./run_demo.sh (Linux)
```

**Features:**
- **Interactive Image Selector:** Tests all 10 CIFAR-10 classes embedded in firmware with zero data-transfer delay.
- **Custom Image Streaming:** Drag-and-drop any external image to stream 3,072 bytes over UART in ~0.26s.
- **Live Layer-by-Layer Animation:** Visualizes each conv/pooling/linear layer as it finishes on hardware.
- **Real-Time Silicon Metrics:** Accurately displays hardware clock cycles, millisecond latency, throughput (FPS), and CPI.
- **Simulator Fallback:** High-fidelity offline fallback mode when the physical FPGA is disconnected.

---

## Methodology

### Core Hardware Proofs & Execution Paradigms

#### 1. SWAR SIMD Register Packing & Fused Sub-Byte Execution
Our framework supports two complementary hardware execution modes for mixed-precision deployment on RISC-V cores:
- **SWAR SIMD Arithmetic:** Utilizes SIMD Within A Register (SWAR) arithmetic to compute multiple low-precision multiply-accumulate operations in parallel within standard 32-bit/64-bit ALU registers, maximizing throughput per clock cycle.
- **Fused Inline Sub-Byte Execution:** Eliminates intermediate unpack buffers (`w_tile[]`) by streaming packed weights directly from ROM/DDR (`weights_packed[]`) in the inner spatial loop. Channel-hoisted bitwidth dispatch extracts 4-bit (2 weights/byte) or 2-bit (4 weights/byte) operands on-the-fly with branchless sign extension (`(int8_t)((byte & 0x0F) << 4) >> 4`), directly reducing dynamic memory load transactions by **~29% to ~50%**.

#### 2. Exact Normalization & BatchNorm Folding
To ensure bit-exact numerical parity between high-level Python simulations and bare-metal C execution:
- **BatchNorm Folding:** BatchNorm scale factors ($\gamma$), offsets ($\beta$), running means ($\mu$), and running variances ($\sigma$) are mathematically folded into convolutional and linear weights prior to quantization:
  $$W_{\text{folded}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad B_{\text{folded}} = (B - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
- **Dynamic Range Extraction:** Per-channel scale projections derive exact $i\_scale$, $w\_scale$, and $o\_scale$ values, ensuring zero numerical drift across deep layers.

#### 3. Granular Sensitivity & Vision Transformer Scaling
- **CNNs:** Filter-level co-optimization ensures sensitive channels retain 8-bit precision while robust channels compress to 4-bit or 2-bit, strictly bounding accuracy drops within $\le 5\%$.
- **Vision Transformers (ViTs):**
  - **LeViT-384:** Serves as our primary mixed-precision ViT benchmark, achieving **97.3% W4/A4 precision** with a **17.05× BOPs reduction**.
  - **Swin-T (CIFAR-100):** Exhibits layer depth scaling of $1/\sqrt{N}$ across 53+ stages, which naturally converges to uniform 8-bit under stringent sensitivity constraints.

---

## Repository Structure

```
REQAP/
├── README.md                           # Master repository documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── models/                             # Pretrained PyTorch checkpoints (.pth)
│
├── quantization_framework/             # Phase 1: PTQ & Sensitivity Search
│   ├── experiments/                    # HRP-Aware greedy bit-width search
│   └── configs/                        # Generated JSON configuration maps
│
├── systolic_sim/                       # Phase 2: Safe-FFD Hardware Simulation
│   └── simulator.py                    # Evaluates exact register fetch reductions
│
├── risc_v_backend/                     # Phase 3: RISC-V C/ASM Execution
│   ├── scripts/                        # Converts models to packed C headers
│   └── firmware/                       # Bare-metal C source code and QEMU Makefile
│
└── VC707_Live_Demo_UI/                 # Phase 4: Interactive Hardware GUI & Telemetry Bridge
    ├── README.md                       # Comprehensive demo & hardware deployment guide
    ├── firmware/                       # Bare-metal C firmware for VC707 (MicroBlaze-V RV64)
    └── python_ui/                      # Streamlit graphical dashboard & UART serial bridge
```

---
