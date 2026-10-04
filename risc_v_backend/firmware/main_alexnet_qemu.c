#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UART_BASE     0x10000000UL
#define UART_THR      (*(volatile uint8_t *)(UART_BASE + 0))
#define UART_LSR      (*(volatile uint8_t *)(UART_BASE + 5))
#define UART_LSR_THRE 0x20U

static void uart_putc(char c) {
    while (!(UART_LSR & UART_LSR_THRE));
    UART_THR = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
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

#ifdef __riscv
static inline uint64_t read_cycles(void) {
    uint32_t h1, l, h2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(h1));
        __asm__ volatile ("rdcycle  %0" : "=r"(l));
        __asm__ volatile ("rdcycleh %0" : "=r"(h2));
    } while (h1 != h2);
    return ((uint64_t)h2 << 32) | l;
}

static inline uint64_t read_instructions(void) {
    uint32_t h1, l, h2;
    do {
        __asm__ volatile ("rdinstreth %0" : "=r"(h1));
        __asm__ volatile ("rdinstret  %0" : "=r"(l));
        __asm__ volatile ("rdinstreth %0" : "=r"(h2));
    } while (h1 != h2);
    return ((uint64_t)h2 << 32) | l;
}
#endif

#include "inference_ops.h"
#include "generated/fashionmnist_test_image.h"

#ifdef BASELINE_MODE
#include "generated/alexnet_baseline_weights.h"
#define MODE_NAME "BASELINE"
#else
#include "generated/alexnet_packed_weights.h"
#define MODE_NAME "PACKED"
#endif

#define RV_LOG(s)

static int8_t fm_A[300000];
static int8_t fm_B[300000];

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_cycles();
    uint64_t start_instrs = read_instructions();
#endif
    
    // conv1
#ifdef BASELINE_MODE
    run_conv2d_8bit(fashionmnist_image_0, fm_A, conv1_0_weights, 227, 227, 1, 55, 55, 96, 11, 11, 4, 0, conv1_0_scale, conv1_0_bias);
#else
    run_conv2d_general(fashionmnist_image_0, fm_A, conv1_0_weights, conv1_0_pos, conv1_0_mask, conv1_0_slots, 227, 227, 1, 55, 55, 96, 11, 11, 4, 0, conv1_0_scale, conv1_0_bias, CONV1_0_MAX_D, CONV1_0_WORDS_PER_FILTER);
#endif
    apply_relu(fm_A, 96 * 55 * 55);
    max_pool_3x3_s2_p0(fm_A, fm_B, 55, 55, 96, 27, 27);
    
    // conv2
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, conv2_0_weights, 27, 27, 96, 27, 27, 256, 5, 5, 1, 2, conv2_0_scale, conv2_0_bias);
#else
    run_conv2d_general(fm_B, fm_A, conv2_0_weights, conv2_0_pos, conv2_0_mask, conv2_0_slots, 27, 27, 96, 27, 27, 256, 5, 5, 1, 2, conv2_0_scale, conv2_0_bias, CONV2_0_MAX_D, CONV2_0_WORDS_PER_FILTER);
#endif
    apply_relu(fm_A, 256 * 27 * 27);
    max_pool_3x3_s2_p0(fm_A, fm_B, 27, 27, 256, 13, 13);
    
    // conv3
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, conv3_0_weights, 13, 13, 256, 13, 13, 384, 3, 3, 1, 1, conv3_0_scale, conv3_0_bias);
#else
    run_conv2d_general(fm_B, fm_A, conv3_0_weights, conv3_0_pos, conv3_0_mask, conv3_0_slots, 13, 13, 256, 13, 13, 384, 3, 3, 1, 1, conv3_0_scale, conv3_0_bias, CONV3_0_MAX_D, CONV3_0_WORDS_PER_FILTER);
#endif
    apply_relu(fm_A, 384 * 13 * 13);
    
    // conv4
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, conv4_0_weights, 13, 13, 384, 13, 13, 384, 3, 3, 1, 1, conv4_0_scale, conv4_0_bias);
#else
    run_conv2d_general(fm_A, fm_B, conv4_0_weights, conv4_0_pos, conv4_0_mask, conv4_0_slots, 13, 13, 384, 13, 13, 384, 3, 3, 1, 1, conv4_0_scale, conv4_0_bias, CONV4_0_MAX_D, CONV4_0_WORDS_PER_FILTER);
#endif
    apply_relu(fm_B, 384 * 13 * 13);
    
    // conv5
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, conv5_0_weights, 13, 13, 384, 13, 13, 256, 3, 3, 1, 1, conv5_0_scale, conv5_0_bias);
#else
    run_conv2d_general(fm_B, fm_A, conv5_0_weights, conv5_0_pos, conv5_0_mask, conv5_0_slots, 13, 13, 384, 13, 13, 256, 3, 3, 1, 1, conv5_0_scale, conv5_0_bias, CONV5_0_MAX_D, CONV5_0_WORDS_PER_FILTER);
#endif
    apply_relu(fm_A, 256 * 13 * 13);
    max_pool_3x3_s2_p0(fm_A, fm_B, 13, 13, 256, 6, 6);
    
    // fc1
#ifdef BASELINE_MODE
    run_linear_8bit(fm_B, fm_A, fc1_weights, 9216, 4096, fc1_scale, fc1_bias);
#else
    run_linear_general(fm_B, fm_A, fc1_weights, fc1_pos, fc1_mask, fc1_slots, 9216, 4096, fc1_scale, fc1_bias, FC1_MAX_D, FC1_WORDS_PER_FILTER);
#endif
    apply_relu(fm_A, 4096);
    
    // fc2
#ifdef BASELINE_MODE
    run_linear_8bit(fm_A, fm_B, fc2_weights, 4096, 4096, fc2_scale, fc2_bias);
#else
    run_linear_general(fm_A, fm_B, fc2_weights, fc2_pos, fc2_mask, fc2_slots, 4096, 4096, fc2_scale, fc2_bias, FC2_MAX_D, FC2_WORDS_PER_FILTER);
#endif
    apply_relu(fm_B, 4096);
    
    // fc3
#ifdef BASELINE_MODE
    run_linear_8bit(fm_B, fm_A, fc3_weights, 4096, 10, fc3_scale, fc3_bias);
#else
    run_linear_general(fm_B, fm_A, fc3_weights, fc3_pos, fc3_mask, fc3_slots, 4096, 10, fc3_scale, fc3_bias, FC3_MAX_D, FC3_WORDS_PER_FILTER);
#endif
    
#ifdef __riscv
    uint64_t end_cycles = read_cycles();
    uint64_t end_instrs = read_instructions();
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;

#ifdef BASELINE_MODE
    uart_puts("===BEGIN_METRICS===\nMODEL=AlexNet Baseline 8-bit\n");
#else
    uart_puts("===BEGIN_METRICS===\nMODEL=AlexNet Packed MQF\n");
#endif

    uart_put_kv64("METRIC_CYCLES",       cycles_delta);
    uart_put_kv64("METRIC_INSTRUCTIONS", instrs_delta);
    uart_put_kv64("METRIC_MEMORY_LOADS", total_memory_loads);
    
    int top1 = 0;
    int32_t top1_score = fm_A[0];
    for (int i = 0; i < 10; i++) {
        if (fm_A[i] > top1_score) {
            top1_score = fm_A[i];
            top1 = i;
        }
        char key[16] = "CLASS_SCORE_X";
        key[12] = '0' + i;
        uart_put_kv32(key, (int32_t)fm_A[i]);
    }
    uart_put_kv32("PRED_TOP1_CLASS",     (int32_t)top1);
    uart_put_kv32("PRED_TOP1_SCORE",     (int32_t)top1_score);
    uart_puts("===END_METRICS===\n");

    volatile uint32_t spin;
    for (spin = 0; spin < 1000000U; spin++);
#else
    printf("\n=== Classification Results ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\n", i, fm_A[i]);
    }
#endif

    return 0;
}
