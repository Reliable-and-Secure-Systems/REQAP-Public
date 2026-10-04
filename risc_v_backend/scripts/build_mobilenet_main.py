import os

with open("risc_v_backend/scripts/mobilenet_pipeline.txt", "r", encoding="utf-16") as f:
    baseline_pipe = f.read()

with open("risc_v_backend/scripts/mobilenet_packed_pipeline.txt", "r", encoding="utf-16") as f:
    packed_pipe = f.read()

main_c_code = f"""/*
 * main_vc707_mobilenet.c
 * ------------------
 * Bare-metal firmware entry point for MobileNet on VC707 FPGA.
 * Supports both Baseline 8-bit and Packed MQF forward passes via compile-time flag.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "swar_mac.h"
#include "hw_metrics.h"

#ifdef __riscv
#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))

static inline void uart_putc(char c) {{
    if (c == '\\n') {{
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\\r';
    }}
    while (UART_TXDATA & 0x80000000);  
    UART_TXDATA = c;
}}

static inline void uart_puts(const char *s) {{
    while (*s) uart_putc(*s++);
}}
#else
static inline void uart_putc(char c) {{
    putchar(c);
}}
static inline void uart_puts(const char *s) {{
    printf("%s", s);
}}
#endif

// Include inference operations
#include "inference_ops.h"
#include "act_packing.h"

// Include Weights and Headers
#ifdef BASELINE_MODE
#include "generated/mobilenet_baseline_weights.h"
#else
#include "generated/mobilenet_packed_weights.h"
#endif

// Include Test Image
#include "generated/cifar_test_image.h"

// Global Feature Map Buffers
int8_t fm_A[300000];
int8_t fm_B[300000];
int8_t res_fm[300000]; // Buffer for residual connections

static void print_result(const char *layer_name, const char *variant, int words, int repeat, int32_t acc) {{
    char buf[128];
    sprintf(buf, "[%s] %s | words=%d x %d | acc=%d\\n", layer_name, variant, words, repeat, acc);
    uart_puts(buf);
}}

static void print_mobilenet_profiler_summary(const char *mode) {{
    char buf[256];
    uart_puts("\\n------------------------------------------------------------\\n");
    sprintf(buf, "  HARDWARE PROFILER RESULTS (%s)\\n", mode);
    uart_puts(buf);
    
    sprintf(buf, "  Total Memory Loads (lw): %llu\\n", (unsigned long long)total_memory_loads);
    uart_puts(buf);
    
    uart_puts("  [Note: Instruction and Cycle counts are only available on RISC-V]\\n");
    uart_puts("------------------------------------------------------------\\n\\n");
}}

#ifdef BASELINE_MODE
static void run_full_mobilenet_baseline_forward_pass(void) {{
    const char *mode_name = "MobileNet Baseline 8-bit";
    printf("\\n=== Starting Full %s Forward Pass ===\\n", mode_name);
    printf("Input Image: CIFAR-10 (3x32x32)\\n");
    total_memory_loads = 0;

{baseline_pipe}

    printf("\\n=== Classification Results ===\\n");
    for (int i = 0; i < 10; i++) {{
        printf("Class %d: %d\\n", i, out_fm[i]);
    }}
    printf("=== End Forward Pass ===\\n\\n");
    print_mobilenet_profiler_summary(mode_name);
}}
#else
const uint8_t dummy_8_act_bits[3] = {{8, 8, 8}};
static void run_full_mobilenet_packed_forward_pass(void) {{
    const char *mode_name = "MobileNet Packed MQF";
    printf("\\n=== Starting Full %s Forward Pass ===\\n", mode_name);
    printf("Input Image: CIFAR-10 (3x32x32)\\n");
    total_memory_loads = 0;

{packed_pipe}

    printf("\\n=== Classification Results ===\\n");
    for (int i = 0; i < 10; i++) {{
        printf("Class %d: %d\\n", i, unpacked_fm[i]);
    }}
    printf("=== End Forward Pass ===\\n\\n");
    print_mobilenet_profiler_summary(mode_name);
}}
#endif

int main(void) {{
    uart_puts("\\n============================================\\n");
    uart_puts("   REQAP-DNN RISC-V INFERENCE ENGINE   \\n");
    uart_puts("============================================\\n");

#ifdef BASELINE_MODE
    run_full_mobilenet_baseline_forward_pass();
#else
    run_full_mobilenet_packed_forward_pass();
#endif

    uart_puts("\\n--------------------------------------------\\n");
    uart_puts("   MOBILENET EXECUTION FINISHED SUCCESSFULLY!  \\n");
    uart_puts("--------------------------------------------\\n");

    return 0;
}}
"""

with open("vc707_deployment/main_vc707_mobilenet.c", "w", encoding="utf-8") as f:
    f.write(main_c_code)

print("Clean main_vc707_mobilenet.c generated.")



