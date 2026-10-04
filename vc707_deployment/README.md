# VC707 Hardware Deployment

This folder contains the self-contained, pre-compiled `.elf` (Executable and Linkable Format) binaries for deploying the MQF-Aware Register Packing optimized models onto the VC707 RISC-V FPGA hardware.

## Purpose
The primary purpose of these binaries is to run the final mixed-precision, heterogeneously packed neural networks (ResNet18, VGG11, and AlexNet) on actual RISC-V hardware to measure the true, cycle-accurate hardware performance and throughput gains achieved by the REQAP-DNN framework. 

Each network has two variants:
- **Baseline**: Standard 8-bit homogeneous quantized compilation for comparison.
- **Packed**: Optimized compilation utilizing the joint W=A packed register scheme.

## Contents
The pre-compiled `.elf` files contain EVERYTHING required to execute inference on the VC707 board. They are completely self-contained and have the following baked directly into the binary:
1. The **Network Architecture** (C implementation)
2. The **Model Weights** (Quantized and Packed configurations)
3. The **Test Images** (For verifying end-to-end correctness)

## Hardware Environment Configuration (Chipyard)

The networks are strictly evaluated on a Xilinx Virtex-7 VC707 FPGA. The underlying System-on-Chip (SoC) was generated using the **UC Berkeley Chipyard Framework**. 

To guarantee reproducibility, we have decoupled the massive hardware generation infrastructure into its own dedicated repository under the RSS organization. You can find the exact hardware generator, boot scripts, and configurations here:
**[RSS-Chipyard (VC707 Clone)](https://github.com/Reliable-and-Secure-Systems/Chipyard/tree/main/fpga/src/main/scala/vc707)**

Our deployment environment relies on this Chipyard output, specifically compiling against its hardware architecture:

- **CPU Core**: Single 64-bit Rocket Core (`rv64imafd`).
- **Chipyard Config**: `new freechips.rocketchip.rocket.WithNHugeCores(1) ++ new chipyard.config.AbstractConfig`
- **Clock Speed**: 100 MHz
- **Main Memory (RAM)**: 1GB DDR3

### Bare-Metal Memory Management
Because these networks execute on bare-metal hardware without an operating system, the deployment relies on a custom `boot1.S` assembly file. This file provides a massive **1MB Memory Stack** directly in the hardware, which is absolutely mandatory to prevent stack-overflow crashes during the heavy forward-pass of deep networks like VGG-11 and ResNet-18. 

Additionally, the binaries utilize direct memory-mapped logging to the VC707 hardware UART at address `0x64000000`, completely bypassing `printf` to ensure hardware stability during execution.

## How to Compile & Use
You can dynamically compile any of the models (Baseline or Packed) directly from source using the provided `Makefile`. The Makefile automatically links the `boot1.S` stack configurations and the `link_uart.ld` linker script.

1. Navigate to this deployment directory.
2. Run `make <target>` (e.g., `make vgg11_packed`, `make resnet18_baseline`, `make alexnet_packed`).
3. To build everything, run `make all`.
4. Flash the generated `.elf` binary into the instruction memory of the RISC-V core using your standard deployment tools (e.g., JTAG, OpenOCD).
5. The program will execute the inference and automatically output the results, memory loads, hardware cycle counts, and **total execution time (in milliseconds)** via the UART interface at `0x64000000`.

## Bare-Metal Standard Library Support
When compiling, you will see `baremetal_libc_stubs.c` linked into the ELF binaries. This file is critical; it provides lightweight, custom implementations of standard C functions (`memset`, `memcpy`, `printf` intercepts) so the bare-metal RISC-V GCC cross-compiler does not crash with "undefined reference" errors.

## Available Binaries & Download Link
Due to GitHub file size constraints (>100MB per binary), the `.elf` files are not tracked in this repository. 

**You can download all 6 compiled binaries from this Google Drive Folder:**
[👉 Download VC707 ELF Binaries Here](https://drive.google.com/drive/folders/13d1KvyH6Lk4hYtuUWnEPAbpVLgpCS3C5?usp=sharing)

The folder contains:
* `alexnet_baseline_vc707.elf` (116.7 MB)
* `alexnet_packed_vc707.elf` (114.3 MB)
* `resnet18_baseline_vc707.elf` (22.5 MB)
* `resnet18_packed_vc707.elf` (12.9 MB)
* `vgg11_baseline_vc707.elf` (56.5 MB)
* `vgg11_packed_vc707.elf` (23.1 MB)
