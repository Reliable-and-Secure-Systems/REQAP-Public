#include "generated/cifar_test_image.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Lightweight logging macro */
#ifdef __riscv
#define RV_LOG(...) do {} while(0)
#define printf(...) do {} while(0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif

// -----------------------------------------------------------------------------
// DIRECT HARDWARE UART LOGGING
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// HARDWARE TIMER (CLINT_MTIME) LOGIC
// -----------------------------------------------------------------------------
#define CLINT_BASE  0x02000000UL
#define CLINT_MTIME (*(volatile uint64_t *)(CLINT_BASE + 0xBFF8))
#define TIMEBASE_HZ 100000UL   // 100 kHz timer clock on VC707

static inline uint64_t mtime_now(void) {
    return CLINT_MTIME; 
}

static inline uint64_t mtime_elapsed_ms(uint64_t start) {
    return ((CLINT_MTIME - start) * 1000) / TIMEBASE_HZ;
}

static void uart_put_kv64(const char *key, uint64_t v) {
    uart_puts(key);
    uart_putc('=');
    char buf[22];
    int i = 21;
    buf[i] = '\0';
    if (v == 0) {
        buf[--i] = '0';
    } else {
        while (v) {
            buf[--i] = '0' + (int)(v % 10);
            v /= 10;
        }
    }
    uart_puts(&buf[i]);
    uart_putc('\n');
}

static void uart_put_kv32(const char *key, int32_t v) {
    uart_puts(key);
    uart_putc('=');
    char buf[12];
    int i = 11;
    buf[i] = '\0';
    int neg = (v < 0);
    uint32_t uv = neg ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
    if (uv == 0) {
        buf[--i] = '0';
    } else {
        while (uv) {
            buf[--i] = '0' + (int)(uv % 10);
            uv /= 10;
        }
    }
    if (neg) buf[--i] = '-';
    uart_puts(&buf[i]);
    uart_putc('\n');
}

#include "hw_metrics.h"
#include "inference_ops.h"
#include "generated/alexnet_test_image.h"

#ifdef BASELINE_MODE
#include "generated/alexnet_baseline_weights.h"
#define MODE_NAME "BASELINE"
#else
#include "generated/alexnet_packed_weights.h"
#define MODE_NAME "PACKED"
#endif

static int8_t fm_A[300000];
static int8_t fm_B[300000];

int main(void) {
    uart_puts("\n\n--------------------------------------------\n");
    uart_puts("   STARTING ALEXNET - VC707 \n");
    uart_puts("--------------------------------------------\n");
    uart_puts("Running forward pass...\n");
    
    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_mcycle();
    uint64_t start_instrs = read_minstret();
    uint64_t start_time   = mtime_now();
#endif

/* Auto-generated alexnet Pipeline (Baseline=True) */
int8_t *unpacked_fm = fm_B;
int8_t *packed_fm = fm_A;
int8_t *tmp;

// Initial state: unpacked_fm has the image.
for (int i=0; i<51529; i++) unpacked_fm[i] = alexnet_image_0[i];

// conv1.0 : 227x227x1 -> 55x55x96 (s=4, p=0)
printf("Running conv1...\\n"); 
#ifdef BASELINE_MODE
run_conv2d_8bit(unpacked_fm, packed_fm, conv1_0_weights,
    227, 227, 1, 55, 55, 96, 11, 11, 0, 4, conv1_0_scale, conv1_0_bias);
#else
run_conv2d_general(unpacked_fm, packed_fm, conv1_0_weights, conv1_0_pos, conv1_0_mask, conv1_0_slots,
    227, 227, 1, 55, 55, 96, 11, 11, 0, 4, conv1_0_d, conv1_0_words, conv1_0_scale, conv1_0_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 290400);

// conv1.2 (MaxPool)
max_pool_2d(unpacked_fm, packed_fm, 55, 55, 96);
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;

// conv2.0 : 27x27x96 -> 27x27x256 (s=1, p=2)
#ifdef BASELINE_MODE
run_conv2d_8bit(unpacked_fm, packed_fm, conv2_0_weights,
    27, 27, 96, 27, 27, 256, 5, 5, 2, 1, conv2_0_scale, conv2_0_bias);
#else
run_conv2d_general(unpacked_fm, packed_fm, conv2_0_weights, conv2_0_pos, conv2_0_mask, conv2_0_slots,
    27, 27, 96, 27, 27, 256, 5, 5, 2, 1, conv2_0_d, conv2_0_words, conv2_0_scale, conv2_0_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 186624);

// conv2.2 (MaxPool)
max_pool_2d(unpacked_fm, packed_fm, 27, 27, 256);
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;

// conv3.0 : 13x13x256 -> 13x13x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_conv2d_8bit(unpacked_fm, packed_fm, conv3_0_weights,
    13, 13, 256, 13, 13, 384, 3, 3, 1, 1, conv3_0_scale, conv3_0_bias);
#else
run_conv2d_general(unpacked_fm, packed_fm, conv3_0_weights, conv3_0_pos, conv3_0_mask, conv3_0_slots,
    13, 13, 256, 13, 13, 384, 3, 3, 1, 1, conv3_0_d, conv3_0_words, conv3_0_scale, conv3_0_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 64896);

// conv4.0 : 13x13x384 -> 13x13x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_conv2d_8bit(unpacked_fm, packed_fm, conv4_0_weights,
    13, 13, 384, 13, 13, 384, 3, 3, 1, 1, conv4_0_scale, conv4_0_bias);
#else
run_conv2d_general(unpacked_fm, packed_fm, conv4_0_weights, conv4_0_pos, conv4_0_mask, conv4_0_slots,
    13, 13, 384, 13, 13, 384, 3, 3, 1, 1, conv4_0_d, conv4_0_words, conv4_0_scale, conv4_0_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 64896);

// conv5.0 : 13x13x384 -> 13x13x256 (s=1, p=1)
#ifdef BASELINE_MODE
run_conv2d_8bit(unpacked_fm, packed_fm, conv5_0_weights,
    13, 13, 384, 13, 13, 256, 3, 3, 1, 1, conv5_0_scale, conv5_0_bias);
#else
run_conv2d_general(unpacked_fm, packed_fm, conv5_0_weights, conv5_0_pos, conv5_0_mask, conv5_0_slots,
    13, 13, 384, 13, 13, 256, 3, 3, 1, 1, conv5_0_d, conv5_0_words, conv5_0_scale, conv5_0_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 43264);

// conv5.2 (MaxPool)
max_pool_2d(unpacked_fm, packed_fm, 13, 13, 256);
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;

// fc1
printf("Running fc1...\\n"); 
#ifdef BASELINE_MODE
run_linear_8bit(unpacked_fm, packed_fm, fc1_weights,
    9216, 4096, fc1_scale, fc1_bias);
#else
run_linear_general(unpacked_fm, packed_fm, fc1_weights, fc1_pos, fc1_mask, fc1_slots,
    9216, 4096, fc1_d, fc1_words, fc1_scale, fc1_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 4096);

// fc2
#ifdef BASELINE_MODE
run_linear_8bit(unpacked_fm, packed_fm, fc2_weights,
    4096, 4096, fc2_scale, fc2_bias);
#else
run_linear_general(unpacked_fm, packed_fm, fc2_weights, fc2_pos, fc2_mask, fc2_slots,
    4096, 4096, fc2_d, fc2_words, fc2_scale, fc2_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;
apply_relu(unpacked_fm, 4096);

// fc3
#ifdef BASELINE_MODE
run_linear_8bit(unpacked_fm, packed_fm, fc3_weights,
    4096, 10, fc3_scale, fc3_bias);
#else
run_linear_general(unpacked_fm, packed_fm, fc3_weights, fc3_pos, fc3_mask, fc3_slots,
    4096, 10, fc3_d, fc3_words, fc3_scale, fc3_bias);
#endif
tmp = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp;


