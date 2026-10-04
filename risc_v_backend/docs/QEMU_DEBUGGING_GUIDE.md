# QEMU Diagnostics & RISC-V Profiling Guide

## 1. Check if QEMU is Running

### PowerShell Commands

```powershell
# Check for active QEMU processes
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue

# More detailed info (PID, CPU%, Memory)
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue | Select-Object Id, Name, CPU, WS, StartTime, Threads

# Count QEMU processes
(Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue | Measure-Object).Count

# Kill all QEMU processes
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue | Stop-Process -Force

# Check if QEMU binary exists
Test-Path "C:\riscv-gcc\bin\qemu-system-riscv32"
```

### Output Interpretation
- **No output**: QEMU not running (good)
- **One or more processes listed**: QEMU still running (may be hung)
- **High CPU% (>50%)**: Firmware likely in infinite loop
- **Growing memory (WS)**: Memory leak or large buffer allocation

---

## 2. Spike Simulator Setup

Spike is the official RISC-V ISA simulator from UC Berkeley and is **better than QEMU for profiling**.

### Check if Spike is Installed

```powershell
# Test if spike is in PATH
spike --version

# If not found, may need to install or add to PATH
which spike   # Linux/WSL
where spike   # PowerShell (if installed)
```

### Install Spike (Windows with WSL/MSYS2)

If you have WSL (Windows Subsystem for Linux):
```bash
sudo apt-get install spike
```

Or if you have MSYS2:
```bash
pacman -S riscv32-unknown-elf-spike
```

### Run Firmware with Spike

```bash
# Basic execution (verbose)
spike -p1 firmware_vgg11_packed.elf

# Run and capture performance metrics
spike --log-commits firmware_vgg11_packed.elf 2> spike_packed.log

# Run with cycle/instruction limit (if firmware hangs)
spike -m0x80000000:0x10000000 firmware_vgg11_packed.elf
```

**Advantage over QEMU**: Spike runs firmware without semihosting; output can be captured via:
- Memory-mapped UART to file
- Direct stdout (if firmware writes to fd=1)
- Profiler dump at exit (rdcycle, rdinstret)

---

## 3. Metrics We're Collecting

### Performance Counters (RISC-V CSRs)

| CSR | Name | What It Measures | Importance |
|-----|------|------------------|-----------|
| **rdcycle** | Cycle Counter | Total clock cycles elapsed | **Critical for latency** |
| **rdinstret** | Instruction Counter | Total instructions executed | **Critical for energy (IPC)** |
| **rdcycleh** / **rdinstreth** | High word (64-bit) | Extension for 32-bit systems | Required for 32-bit RISC-V |

### Code-Level Metrics (Already Collected)

| Metric | Where Tracked | Value (Packed) | Value (Baseline) | Significance |
|--------|---------------|---|---|---|
| **Memory Loads (lw)** | `total_memory_loads` in inference_ops.c | 147.66M | 171.68M | Register reads from weight banks |
| **Layer Activations** | Range checks in main_inference.c | All present | All present | Data distribution, overflow detection |
| **Unit Test Results** | features.0, .25, classifier.0 | All pass | All pass | Correctness verification |

### Memory Architecture Metrics

| Metric | How to Measure | Why It Matters |
|--------|-----------------|---|
| **Memory Bandwidth** | (Total Data Bytes) / (Cycle Count) | FPGA I/O pin count |
| **Cache Misses** | Not directly available in ISS; estimate from lw count | FPGA on-chip memory size |
| **Register Pressure** | (Live registers) / (Register file size) | FPGA slice/LUT utilization |
| **Memory Latency** | Cycle stall from load-to-use dependency | FPGA pipeline depth |

---

## 4. Understanding the RISC-V → FPGA Path

### Current Phase: Software Profiling (ISA Simulation)
```
C Code (main_inference.c)
    ↓
Cross-Compile to RISC-V ISA
    ↓
Run on Simulator (QEMU or Spike)
    ↓
Collect Metrics: cycles, instructions, memory loads
    ↓
Analyze Efficiency (packed vs. baseline)
```

**Goal**: Understand what the algorithm *needs* in terms of compute and memory.

### Next Phase: Hardware Implementation (FPGA)
```
Performance Metrics from Simulation
    ↓
Design FPGA Architecture:
  - Datapath width (how many multipliers?)
  - Memory depth (how many weights on-chip?)
  - Register file size (how many live variables?)
  - Compute units (parallel MACs?)
    ↓
Synthesize to FPGA (Vivado, ISE, etc.)
    ↓
Physical Constraints:
  - Slice/LUT usage
  - BRAM usage
  - DSP usage (multiply-accumulate blocks)
  - Power consumption
    ↓
Final FPGA Bitstream → Deploy to Hardware
```

---

## 5. Why These Metrics Matter for FPGA

### Cycle Count
- **RISC-V ISS gives**: Total cycles at "1 GHz virtual frequency"
- **FPGA design**: Real frequency depends on synthesis constraints (e.g., 50 MHz, 100 MHz, 200 MHz)
- **Latency**: `latency_ms = (cycle_count / fpga_frequency_hz) × 1000`

### Instruction Count
- **RISC-V ISS gives**: Total dynamic instructions (load, store, add, multiply, etc.)
- **FPGA design**: Each instruction maps to logic gates or DSP blocks
- **Energy**: `power ∝ (instruction_count × voltage²) / frequency`
- **Packed vs. baseline**: Fewer instructions = lower power on FPGA

### Memory Loads (lw)
- **RISC-V ISS gives**: 147.66M vs. 171.68M words
- **FPGA design**:
  - If **1 lw per cycle**: ~15% fewer cycle stalls for packed model
  - Memory bandwidth requirement: `bw = (load_count × word_width) / cycle_count`
  - **For packed**: `bw = (147.66M × 16 bits) / cycle_count`
  - **For baseline**: `bw = (171.68M × 16 bits) / cycle_count`
  - Directly affects BRAM usage and HBM (high-bandwidth memory) requirements

### Register Size
- **Why it matters**: 
  - Each activation/intermediate result needs a register
  - FPGA has finite distributed RAM or registered logic
  - Packed model reduces weight counts → smaller register file needed
  - Estimate: If 32 live registers, baseline needs 32×16 bits = 512 bits; packed might need 32×12 bits = 384 bits

---

## 6. Recommended Next Steps

### Option A: Fix QEMU Semihosting (If you prefer QEMU)
```powershell
# Try with reduced output buffering
qemu-system-riscv32 -machine virt -bios none -kernel firmware_vgg11_packed.elf `
  -nographic -serial mon:stdio `
  -semihosting -semihosting-config enable=on,target=native,chardev=ser0
```

### Option B: Switch to Spike (Recommended)
```bash
# If Spike available:
spike firmware_vgg11_packed.elf > spike_packed_output.txt 2>&1
spike firmware_vgg11_baseline.elf > spike_baseline_output.txt 2>&1

# Extract metrics from output and log files
```

### Option C: Modify Firmware for Direct Output (No Semihosting)
Add memory-mapped UART output:
```c
// Write to QEMU virt UART @ 0x10010000
volatile uint8_t *uart = (uint8_t *)0x10010000;
*uart = 'H';  // Print 'H'
*uart = 'i';  // Print 'i'
```

---

## 7. Profiling Flow

```
┌─────────────────────────────────────────┐
│ Compile Both Models for RISC-V          │
│ (packed & baseline with -O2)            │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│ Run on Spike/QEMU                       │
│ Capture: cycle_count, instr_count       │
│ Already have: memory_loads              │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│ Extract Metrics (See Example Below)     │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│ Compare Packed vs. Baseline             │
│ - Cycle reduction                       │
│ - Instruction reduction                 │
│ - Memory load reduction (done: 14%)     │
│ - Energy estimate (cycles × instr)      │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│ Design FPGA Datapath Based on Metrics   │
│ - Parallelism (# of MACs)               │
│ - Memory hierarchy (on-chip vs. off)    │
│ - Register file size                    │
└─────────────────────────────────────────┘
```

---

## 8. Example Metrics Table (What We'll Extract)

Once QEMU/Spike runs successfully:

```
Model                    | Cycles    | Instructions | Mem Loads | IPC  | Energy*
-------------------------|-----------|--------------|-----------|------|--------
Packed MQF (RISC-V)      | TBD       | TBD          | 147.66M   | TBD  | TBD
Baseline 8-bit (RISC-V)  | TBD       | TBD          | 171.68M   | TBD  | TBD
Improvement              | TBD%      | TBD%         | +14.0%    | TBD% | TBD%

* Energy estimate = Cycles × Instructions (proxy; actual depends on FPGA frequency, voltage)
```

---

## 9. References

- **Spike ISA Simulator**: https://github.com/riscv-software-consortium/riscv-isa-sim
- **QEMU RISC-V**: https://wiki.qemu.org/Documentation/Platforms/RISC-V
- **RISC-V Specs**: https://riscv.org/technical/specifications/
- **FPGA Design Flow**: Xilinx Vivado, Intel Quartus documentation
