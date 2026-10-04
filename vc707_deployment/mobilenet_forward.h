#include <stdlib.h>
#include "inference_ops.h"
#include "act_packing.h"
#include "generated/cifar_test_image.h"

// Ping-Pong Buffers for intermediate feature maps
static int8_t fm_A[65536];
static int8_t fm_B[65536];

#ifdef __riscv
extern void uart_print_dec(int64_t v);
#else
#include <stdio.h>
#define uart_print_dec(v) printf("%lld", (long long)(v))
#endif

static void print_mobilenet_profiler_summary(const char *mode_name,
                                           uint64_t total_loads,
                                           uint64_t cycles,
                                           uint64_t instrs,
                                           uint64_t time_ms)
{
    uart_puts("------------------------------------------------------------\n");
    uart_puts("  HARDWARE PROFILER RESULTS (");
    uart_puts(mode_name);
    uart_puts(")\n");
    
    uart_puts("  Total Memory Loads (lw): ");
    uart_print_dec(total_loads);
    uart_puts("\n");
#ifdef __riscv
    uart_puts("  Total Instruction Count: ");
    uart_print_dec(instrs);
    uart_puts("\n");
    
    uart_puts("  Total CPU Cycle Count  : ");
    uart_print_dec(cycles);
    uart_puts("\n");
    
    uart_puts("  Total Execution Time   : ");
    uart_print_dec(time_ms);
    uart_puts(" ms\n");
#else
    uart_puts("  [Note: Instruction and Cycle counts are only available on RISC-V]\n");
#endif
    uart_puts("------------------------------------------------------------\n");
}

void run_full_mobilenet_baseline_forward_pass(void)
{
    const char *mode_name = "MobileNet Baseline 8-bit";

    printf("\n=== Starting Full %s Forward Pass ===\n", mode_name);
    printf("Input Image: CIFAR-10 (3x32x32)\n");

    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_mcycle();
    uint64_t start_instrs = read_minstret();
    uint64_t start_time   = mtime_now();
#endif

    // Load image into fm_A
    for (int i = 0; i < 3072; i++) {
        fm_A[i] = cifar_image_0[i];
    }
    
    // START PIPELINE
PIPELINE_MARKER
    // END PIPELINE
    
    printf("\n=== Classification Results ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\n", i, out_fm[i]);
    }
    printf("=== End Forward Pass ===\n\n");

#ifdef __riscv
    uint64_t end_cycles = read_mcycle();
    uint64_t end_instrs = read_minstret();
    uint64_t end_time   = mtime_now();
#endif

#ifdef __riscv
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;
    uint64_t time_ms      = mtime_elapsed_ms(start_time);
#else
    uint64_t cycles_delta = 0ULL;
    uint64_t instrs_delta = 0ULL;
    uint64_t time_ms      = 0ULL;
#endif

    print_mobilenet_profiler_summary(mode_name,
                                 total_memory_loads,
                                 cycles_delta,
                                 instrs_delta,
                                 time_ms
                                 );
}

void run_full_mobilenet_packed_forward_pass(void)
{
    // TO BE IMPLEMENTED
    printf("Packed forward pass not implemented yet.\n");
}
