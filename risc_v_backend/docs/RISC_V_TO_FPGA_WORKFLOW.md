# RISC-V to FPGA: Complete Workflow & Metrics Collection

## TL;DR Quick Summary

| Question | Answer |
|----------|--------|
| **Check QEMU running?** | `Get-Process qemu-system-riscv32` (see examples below) |
| **Use Spike instead?** | Spike not installed; could install via WSL/MSYS2, but not necessary |
| **Get metrics without QEMU?** | ✅ **Use PC results + direct firmware modification** (recommended) |
| **What metrics for FPGA?** | Cycles, Instructions, Memory loads (all can be extracted from PC or modified firmware) |
| **How does this go to FPGA?** | Simulation metrics → FPGA datapath design → Synthesis → Bitstream → Hardware |

---

## Part 1: Check QEMU Status

### Commands to Use

```powershell
# ===== LIST ALL RUNNING QEMU PROCESSES =====
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue

# ===== SHOW DETAILED INFO (CPU%, Memory, etc.) =====
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue | `
  Select-Object Id, Name, CPU, @{Name='Mem(MB)';Expression={[math]::Round($_.WS/1MB,2)}}, StartTime

# ===== COUNT HOW MANY QEMU PROCESSES =====
(Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue).Count

# ===== KILL ALL QEMU PROCESSES =====
Get-Process qemu-system-riscv32 -ErrorAction SilentlyContinue | Stop-Process -Force

# ===== CHECK IF QEMU BINARY EXISTS =====
Test-Path "C:\Program Files\QEMU\qemu-system-riscv32.exe"
```

### What to Look For

```
✅ GOOD: No output (QEMU not running)
⚠️  BAD: CPU% > 100 or Memory > 500MB (firmware hung in infinite loop)
❌ CRITICAL: Multiple QEMU processes (previous runs not cleaned up)
```

---

## Part 2: Alternative Solution (Recommended)

Since QEMU semihosting is unreliable, we have **3 practical options**:

### Option A: Use PC Results + Simple RISC-V Estimation
**Time**: 5 minutes | **Effort**: Minimal | **Accuracy**: Good enough for FPGA design

We **already have** from PC execution:
- ✅ Memory loads: 147.66M (packed) vs 171.68M (baseline) = **14% reduction**
- ✅ Both models predict same class (accuracy verified)
- ✅ Layer-by-layer activation ranges

**Estimate cycle/instruction counts** using:
- RISC-V ISA specs: Each instruction type has latency (add=1 cycle, multiply=3 cycles, load=2 cycles)
- Count instructions by type: `addi`, `lw`, `mul`, `sw`, etc.
- Formula: `total_cycles ≈ (fast_instructions × 1) + (multiply × 3) + (load/store × 2)`

**Advantage**: No simulator needed, data already in hand  
**Disadvantage**: Estimates, not exact measurements

---

### Option B: Modify Firmware to Write Metrics to File (Best)
**Time**: 20 minutes | **Effort**: Small code change | **Accuracy**: Perfect

Instead of relying on semihosting `printf()`, write metrics directly to a file handle or memory region.

#### Modified `main_inference.c` Approach:

```c
// Add near the end of firmware (before exit)
#ifdef __riscv
static void dump_metrics_to_file(const char *filename, 
                                  uint64_t cycles, 
                                  uint64_t instructions,
                                  uint32_t memory_loads) {
    // Use minimal file I/O (avoid large buffering)
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "CYCLES,%llu\n", cycles);
        fprintf(f, "INSTRUCTIONS,%llu\n", instructions);
        fprintf(f, "MEMORY_LOADS,%u\n", memory_loads);
        fclose(f);
    }
}

// In main() at the end:
uint64_t end_cycles = read_cycles();
uint64_t end_instructions = read_instructions();
dump_metrics_to_file("metrics_packed.csv", end_cycles, end_instructions, total_memory_loads);
```

**How it works**:
1. Firmware runs in QEMU with semihosting enabled
2. Instead of printing 1000s of lines, just writes 3 lines to a CSV file
3. QEMU semihosting can handle small file writes reliably
4. Extract metrics from generated CSV file

**Steps to implement**:
1. Modify `main_inference.c` to add `dump_metrics_to_file()` function
2. Recompile firmware: `make qemu-vgg11-packed`
3. Run: `qemu-system-riscv32 ... firmware_vgg11_packed.elf`
4. Check generated file: `metrics_packed.csv`
5. Parse and compare

---

### Option C: Install Spike Simulator (Advanced but Reliable)
**Time**: 30 minutes | **Effort**: Installation + learning curve | **Accuracy**: Perfect

Spike is the official RISC-V ISA simulator. It's more reliable than QEMU for profiling.

#### Install Spike on Windows (via WSL)

If you have WSL installed:
```bash
# Inside WSL terminal
wsl
sudo apt-get update
sudo apt-get install -y spike

# Then run from WSL
spike firmware_vgg11_packed.elf
```

#### Or Install via MSYS2 (if you have it)
```bash
pacman -S riscv32-unknown-elf-spike
```

#### Use Spike to Profile
```bash
# Run and capture all output
spike firmware_vgg11_packed.elf > spike_packed.txt 2>&1

# Or with instruction tracing (very verbose)
spike --log-commits firmware_vgg11_packed.elf 2> spike_packed_trace.log 1> spike_packed.txt
```

**Advantages**:
- No semihosting needed (works standalone)
- More reliable than QEMU
- Built-in performance counting
- Smaller memory footprint

**Disadvantage**: Need WSL or MSYS2

---

## Part 3: What Metrics Mean for FPGA

### The Simulation → FPGA Flow

```
┌──────────────────────────────────────────┐
│  1. RISC-V Simulation (PC or Spike)      │
│     • Cycle count: How many clock ticks   │
│     • Instr count: How many operations    │
│     • Memory loads: Register accesses     │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  2. Extract Design Parameters            │
│     Cycles per frame: 1,234,567          │
│     Instr per frame: 5,678,901           │
│     Loads per frame: 147,660,800         │
│     IPC (Instr/Cycle): 4.6               │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  3. Design FPGA Architecture             │
│     • Frequency: 100 MHz (design choice) │
│     • Latency: cycles / frequency        │
│     •   = 1,234,567 / 100M = 12.3 ms    │
│     • Throughput: fps / latency          │
│     •   = 30 fps / 12.3 ms = 2.4 frames │
│     • Datapath: # of MACs, parallelism   │
│     • Memory: On-chip BRAM size          │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  4. Synthesis & Place & Route (P&R)      │
│     Tool: Vivado, Quartus, etc.          │
│     Output: Utilization Report           │
│     • LUT usage: X%                       │
│     • BRAM usage: Y%                      │
│     • DSP usage: Z%                       │
│     • Power: W watts                      │
│     • Max frequency: F MHz                │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  5. Generate Bitstream                   │
│     • .bit file for Xilinx                │
│     • .sof file for Intel/Altera          │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  6. Deploy to Hardware                   │
│     • Program FPGA via USB, JTAG, etc.   │
│     • Run inference on real hardware      │
│     • Measure real power, latency, etc.   │
└──────────────────────────────────────────┘
```

---

## Part 4: Specific Metrics You Need

### 1. Cycle Count
**What it is**: Total clock ticks to execute entire VGG-11 inference  
**From simulation**: Read `rdcycle` CSR (already in firmware)  
**Example**:
```
Packed:   500M cycles
Baseline: 600M cycles
Packed is 16.7% faster
```
**For FPGA**: 
- `Latency (ms) = cycles / (frequency_MHz × 1,000,000)`
- At 100 MHz: packed = 5 ms, baseline = 6 ms

### 2. Instruction Count
**What it is**: Total number of RISC-V instructions executed  
**From simulation**: Read `rdinstret` CSR (already in firmware)  
**Example**:
```
Packed:   1.2B instructions
Baseline: 1.5B instructions
Packed uses 20% fewer instructions
```
**For FPGA**:
- **IPC** (Instructions Per Cycle) = `instructions / cycles`
  - Packed IPC = 1.2B / 500M = 2.4 instr/cycle
  - Baseline IPC = 1.5B / 600M = 2.5 instr/cycle
- **Energy proxy**: More instructions → more power
  - `Energy ∝ Instructions × Voltage² / Frequency`

### 3. Memory Loads (Already Have!)
**What it is**: Number of `lw` (load word) instructions to fetch weights  
**From code**: `total_memory_loads` in inference_ops.c  
**Results**:
```
Packed:   147,660,800 loads
Baseline: 171,679,744 loads
Packed saves 24M loads = 14% reduction
```
**For FPGA**:
- **Memory bandwidth requirement**: `(loads × word_width) / cycles`
  - Packed: (147.66M × 16 bits) / 500M cycles = ~4.7 Gb/s
  - Baseline: (171.68M × 16 bits) / 600M cycles = ~4.6 Gb/s
- **On-chip memory (BRAM)**: Packed model fits better due to compression
- **Off-chip bandwidth**: Fewer loads = lower power to external DRAM

### 4. Register File Size
**What it is**: Maximum live variables at any point in execution  
**Estimate**: Count intermediate activation buffers in code
  - Input: 3×32×32 = 3,072 bytes
  - Layer outputs: largest is features.15 (16×8×8×512 = 524,288 bytes)
  - Max ~10 MB total live data
**For FPGA**:
- Packed model: Weights compressed 6-8×, so less memory needed
- Baseline: Full 8-bit weights, larger footprint
- FPGA BRAM allocation: Packed uses less, baseline uses more

### 5. Power Consumption (Indirect)
**Formula**: `Power = (Frequency × Voltage²) × (Switched Capacitance)`
**Switched Capacitance ∝ Number of Operations**
- Fewer instructions → Lower power
- Fewer memory loads → Lower power
**Example**:
```
Packed energy  = 100M freq × 1V² × 0.8 (20% fewer ops) = Relative power 0.8
Baseline energy = 100M freq × 1V² × 1.0 (baseline)      = Relative power 1.0
Packed saves 20% power vs baseline
```

---

## Part 5: Why FPGA Instead of GPU/CPU?

| Aspect | GPU | CPU | FPGA |
|--------|-----|-----|------|
| **Power** | 100-300W | 50-150W | 5-50W ⭐ |
| **Latency** | 10-100 ms | 5-50 ms | **1-10 ms** ⭐ |
| **Throughput** | 1000s TFLOPS | 100s GFLOPS | Custom, usually 10-1000 GFLOPS |
| **Cost** | $300-3000 | $300-1500 | $100-500 ⭐ |
| **For embedded inference** | Overkill | Overkill | Perfect ⭐ |
| **Programming** | CUDA/OpenCL | C++ | HDL (Verilog/VHDL) ⚠️ |

**Why your thesis uses FPGA**: You're designing an **embedded CNN accelerator** for IoT/edge devices where:
- Power budget is tight (battery-powered)
- Latency must be deterministic (hard real-time)
- Cost per unit matters (many devices deployed)

The metrics you're collecting (cycle count, memory bandwidth, register pressure) **directly inform FPGA design decisions**:
- Packed model → 14% fewer memory loads → **Smaller on-chip memory, lower power**
- Fewer instructions → **Fewer ALUs/DSPs needed, smaller area**
- Cycle count → **Real-time guarantees**

---

## Part 6: Recommended Action Plan

### **TODAY (Right Now)**

1. ✅ **Option A** (5 min): Use PC results
   - You already have memory loads: 14% improvement (packed)
   - You already have accuracy: Both models predict same class
   - Estimate cycles using ISA manual (rough but useful)

2. ✅ **Option B** (20 min): Modify firmware for reliable output
   - Small code change to dump metrics to CSV
   - Recompile and run QEMU with minimal semihosting overhead
   - Extract exact metrics

3. ❌ **Option C** (30+ min): Install Spike
   - Better long-term, but not necessary right now
   - Defer if you're on schedule deadline

### **FPGA Design Phase (Next Step)**

Once you have cycle/instruction counts:
1. Create comparison table:
   ```
   Metric          | Packed MQF | Baseline 8-bit | Improvement
   Cycles          | TBD        | TBD            | TBD
   Instructions    | TBD        | TBD            | TBD
   Memory Loads    | 147.66M    | 171.68M        | +14.0%
   IPC             | TBD        | TBD            | TBD
   ```

2. Design FPGA datapath based on:
   - Cycle count → Latency requirement
   - Instructions → Number of execution units
   - Memory loads → BRAM/bandwidth requirement
   - Register count → On-chip register file size

3. Synthesize and measure real hardware metrics

---

## Code Examples to Implement

### Add This to `main_inference.c`:

```c
// At the top of main()
uint64_t start_cycles, start_instructions;
#ifdef __riscv
start_cycles = read_cycles();
start_instructions = read_instructions();
#endif

// At the end of main() before return
#ifdef __riscv
uint64_t end_cycles = read_cycles();
uint64_t end_instructions = read_instructions();
printf("METRIC_CYCLES=%llu\n", end_cycles - start_cycles);
printf("METRIC_INSTRUCTIONS=%llu\n", end_instructions - start_instructions);
printf("METRIC_MEMORY_LOADS=%u\n", total_memory_loads);
#endif
```

Then run and parse output:
```bash
qemu-system-riscv32 ... firmware_vgg11_packed.elf 2>&1 | grep METRIC
# Output:
# METRIC_CYCLES=500123456
# METRIC_INSTRUCTIONS=1200567890
# METRIC_MEMORY_LOADS=147660800
```

---

## Summary: What To Do Now

| Step | Action | Time | Result |
|------|--------|------|--------|
| 1 | Implement Option B (firmware modification) | 20 min | Get exact cycle/instr counts from QEMU |
| 2 | Recompile and run | 5 min | CSV output file with metrics |
| 3 | Extract and create comparison table | 10 min | Full metrics for FPGA design |
| 4 | Proceed to FPGA synthesis | N/A | Use metrics to design datapath |

**Total time**: ~35 minutes to have all metrics ready for FPGA design.

---

**Next: Shall I modify your firmware to implement Option B? (Recommended)**
