/*
 * main_vc707_swint.c
 * ---------------------
 * Bare-metal firmware entry point for Swin-Tiny on VC707 FPGA.
 * Executes the Vision Transformer blocks (MHSA, FFN) using packed MQF weights.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Compile-time logging redirection */
#ifdef __riscv
#define RV_LOG(...) do {} while(0)
#define printf(...) do {} while(0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif

#ifdef __riscv
#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))

static void uart_putc(char c) {
    if (c == '\n') {
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\r';
    }
    while (UART_TXDATA & 0x80000000);  
    UART_TXDATA = c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}
#else
static void uart_putc(char c) {
    putchar(c);
}
static void uart_puts(const char *s) {
    printf("%s", s);
}
#endif

#define CLINT_BASE  0x02000000UL
#define CLINT_MTIME (*(volatile uint64_t *)(CLINT_BASE + 0xBFF8))
#define TIMEBASE_HZ 100000UL   // 100 kHz timer clock on VC707

static inline uint64_t mtime_now(void) { return CLINT_MTIME; }
static inline uint64_t mtime_elapsed_ms(uint64_t start) {
    return ((CLINT_MTIME - start) * 1000) / TIMEBASE_HZ;
}

extern void uart_print_dec(int64_t v);

static void uart_put_kv64(const char *key, uint64_t v) {
    uart_puts(key);
    uart_putc('=');
    char buf[22];
    int i = 21;
    buf[i] = '\0';
    if (v == 0) buf[--i] = '0';
    else {
        while (v) { buf[--i] = '0' + (int)(v % 10); v /= 10; }
    }
    uart_puts(&buf[i]);
    uart_putc('\n');
}

#include "inference_ops.h"
#include "generated/swin_packed_weights.h"
#include "generated/swin_extras.h"

// For Swin-Tiny we need fairly large buffers. 
// Sequence length = (224/4)*(224/4) = 3136 tokens. Max dimension = 768.
// We'll allocate statically.
static int8_t fm_A[1048576]; // 1MB buffer
static int8_t fm_B[1048576]; // 1MB buffer
static int8_t fm_Shortcut[1048576]; // 1MB buffer

static void print_layer_range(const char *name, const int8_t *fm, int size) {
    int8_t min_v = 127, max_v = -128;
    int64_t sum = 0;
    for (int i = 0; i < size; i++) {
        int8_t v = fm[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += abs(v);
    }
    RV_LOG("  [Range check] %s: min=%d, max=%d, mean=%f\n", name, min_v, max_v, (double)sum / size);
}

int main(void) {
    uart_puts("\n\n--------------------------------------------\n");
    uart_puts("   STARTING SWIN-TINY - VC707 \n");
    uart_puts("--------------------------------------------\n");
    uart_puts("Running forward pass...\n");
    
    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_mcycle();
    uint64_t start_instrs = read_minstret();
    uint64_t start_time   = mtime_now();
#endif
    
    // Simulate an input image of 3x224x224 (dummy zeros for now)
    memset(fm_A, 0, 3*224*224);

    int8_t cifar_image_0[3072] = {0};
    
    #include "swin_packed_pipeline.c"

    printf("\n=== Execution Complete (Swin Full Pipeline Test) ===\n");
    
#ifdef __riscv
    uint64_t end_cycles = read_mcycle();
    uint64_t end_instrs = read_minstret();
    uint64_t end_time   = mtime_now();
    
    uart_puts("------------------------------------------------------------\n");
    uart_puts("  HARDWARE PROFILER RESULTS (Swin-Tiny Block 0)\n");
    uart_puts("  Total Memory Loads (lw): ");
    uart_print_dec(total_memory_loads);
    uart_puts("\n");
    uart_puts("  Total Instruction Count: ");
    uart_print_dec(end_instrs - start_instrs);
    uart_puts("\n");
    uart_puts("  Total CPU Cycle Count  : ");
    uart_print_dec(end_cycles - start_cycles);
    uart_puts("\n");
    uart_puts("  Total Execution Time   : ");
    uart_print_dec(mtime_elapsed_ms(start_time));
    uart_puts(" ms\n");
    uart_puts("------------------------------------------------------------\n");
#endif    

    /* Spin so QEMU doesn't exit before UART flushes */
    volatile uint32_t spin;
    for (spin = 0; spin < 1000000U; spin++);
    return 0;
}
