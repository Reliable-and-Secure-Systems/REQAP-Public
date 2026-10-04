"""
config.py
---------
Central System Configuration for VC707 FPGA Demonstration Platform.
Modify clock frequencies, target architecture parameters, and serial defaults here.
"""

# Hardware Clock Frequency in Hz (Default: 100 MHz for Xilinx VC707 Virtex-7 SoC)
FPGA_CLK_FREQ_HZ = 100_000_000

# Default UART Serial Communication Parameters
DEFAULT_BAUD_RATE = 115200
SUPPORTED_BAUD_RATES = [115200, 921600, 57600]

# Supported CIFAR-10 Target Categories
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Hardware Baseline Specifications for Presentation
BASELINE_TARGET_LATENCY_MS = 14.20
BASELINE_TARGET_CYCLES = 1_420_500
BASELINE_TARGET_INSTRS = 1_120_400
BASELINE_TARGET_FPS = 70.4
BASELINE_TARGET_CPI = 1.27
