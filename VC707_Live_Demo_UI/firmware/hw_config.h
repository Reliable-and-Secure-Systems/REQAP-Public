#ifndef HW_CONFIG_H
#define HW_CONFIG_H

#include <stdint.h>

/* ==============================================================================
 * Central Hardware Clock Configuration for Xilinx VC707 RISC-V SoC
 * ==============================================================================
 * Modify FPGA_CLK_FREQ_HZ if the FPGA bitstream is synthesized at a different
 * clock frequency (e.g., 50 MHz, 100 MHz, 150 MHz, 200 MHz).
 */
#define FPGA_CLK_FREQ_HZ    100000000ULL  /* 100 MHz (100,000,000 Hz) */

#endif /* HW_CONFIG_H */
