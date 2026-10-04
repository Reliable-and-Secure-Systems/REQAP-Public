/*
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

static inline void uart_putc(char c) {
    if (c == '\n') {
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\r';
    }
    while (UART_TXDATA & 0x80000000);  
    UART_TXDATA = c;
}

static inline void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}
#else
static inline void uart_putc(char c) {
    putchar(c);
}
static inline void uart_puts(const char *s) {
    printf("%s", s);
}
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
int8_t res_fm[300000];
int8_t current_test_image[3072]; // Buffer for residual connections

static void print_result(const char *layer_name, const char *variant, int words, int repeat, int32_t acc) {
    char buf[128];
    sprintf(buf, "[%s] %s | words=%d x %d | acc=%d\n", layer_name, variant, words, repeat, acc);
    uart_puts(buf);
}

static void print_mobilenet_profiler_summary(const char *mode) {
    char buf[256];
    uart_puts("\n------------------------------------------------------------\n");
    sprintf(buf, "  HARDWARE PROFILER RESULTS (%s)\n", mode);
    uart_puts(buf);
    
    sprintf(buf, "  Total Memory Loads (lw): %llu\n", (unsigned long long)total_memory_loads);
    uart_puts(buf);
    
    uart_puts("  [Note: Instruction and Cycle counts are only available on RISC-V]\n");
    uart_puts("------------------------------------------------------------\n\n");
}

#ifdef BASELINE_MODE
static int run_full_mobilenet_baseline_forward_pass(void) {
    const char *mode_name = "MobileNet Baseline 8-bit";
    printf("\n=== Starting Full %s Forward Pass ===\n", mode_name);
    printf("Input Image: CIFAR-10 (3x32x32)\n");
    total_memory_loads = 0;

/* Auto-generated MobileNet Baseline Pipeline */
int8_t *in_fm = fm_A;
int8_t *out_fm = fm_B;
int8_t *res_fm_ptr = res_fm;
int8_t *tmp;

// Initial state: in_fm has the image.
for (int i=0; i<3072; i++) in_fm[i] = current_test_image[i];

// features.0.0 : 32x32x3 -> 32x32x32 (s=1, p=1)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_0_0_weights,
    32, 32, 3, 32, 32, 32, 3, 3, 1, 1, features_0_0_scale, features_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_0_0_weights, features_0_0_pos, features_0_0_mask, features_0_0_slots,
    NULL, NULL,
    32, 32, 3, 32, 32, 32, 3, 3, 1, 1, features_0_0_d, features_0_0_words, features_0_0_scale, features_0_0_bias);
#endif
apply_relu(out_fm, 32768);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.1.conv.0.0 : 32x32x32 -> 32x32x32 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_1_conv_0_0_weights,
    32, 32, 32, 32, 32, 3, 3, 1, 1, features_1_conv_0_0_scale, features_1_conv_0_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_1_conv_0_0_weights, features_1_conv_0_0_pos, features_1_conv_0_0_mask, features_1_conv_0_0_slots,
    NULL, NULL,
    32, 32, 32, 32, 32, 3, 3, 1, 1, features_1_conv_0_0_d, features_1_conv_0_0_words, features_1_conv_0_0_scale, features_1_conv_0_0_bias);
#endif
apply_relu(out_fm, 32768);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.1.conv.1 : 32x32x32 -> 32x32x16 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_1_conv_1_weights,
    32, 32, 32, 32, 32, 16, 1, 1, 0, 1, features_1_conv_1_scale, features_1_conv_1_bias);
#else
run_conv2d_general(in_fm, out_fm, features_1_conv_1_weights, features_1_conv_1_pos, features_1_conv_1_mask, features_1_conv_1_slots,
    NULL, NULL,
    32, 32, 32, 32, 32, 16, 1, 1, 0, 1, features_1_conv_1_d, features_1_conv_1_words, features_1_conv_1_scale, features_1_conv_1_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.2.conv.0.0 : 32x32x16 -> 32x32x96 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_2_conv_0_0_weights,
    32, 32, 16, 32, 32, 96, 1, 1, 0, 1, features_2_conv_0_0_scale, features_2_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_2_conv_0_0_weights, features_2_conv_0_0_pos, features_2_conv_0_0_mask, features_2_conv_0_0_slots,
    NULL, NULL,
    32, 32, 16, 32, 32, 96, 1, 1, 0, 1, features_2_conv_0_0_d, features_2_conv_0_0_words, features_2_conv_0_0_scale, features_2_conv_0_0_bias);
#endif
apply_relu(out_fm, 98304);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.2.conv.1.0 : 32x32x96 -> 32x32x96 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_2_conv_1_0_weights,
    32, 32, 96, 32, 32, 3, 3, 1, 1, features_2_conv_1_0_scale, features_2_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_2_conv_1_0_weights, features_2_conv_1_0_pos, features_2_conv_1_0_mask, features_2_conv_1_0_slots,
    NULL, NULL,
    32, 32, 96, 32, 32, 3, 3, 1, 1, features_2_conv_1_0_d, features_2_conv_1_0_words, features_2_conv_1_0_scale, features_2_conv_1_0_bias);
#endif
apply_relu(out_fm, 98304);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.2.conv.2 : 32x32x96 -> 32x32x24 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_2_conv_2_weights,
    32, 32, 96, 32, 32, 24, 1, 1, 0, 1, features_2_conv_2_scale, features_2_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_2_conv_2_weights, features_2_conv_2_pos, features_2_conv_2_mask, features_2_conv_2_slots,
    NULL, NULL,
    32, 32, 96, 32, 32, 24, 1, 1, 0, 1, features_2_conv_2_d, features_2_conv_2_words, features_2_conv_2_scale, features_2_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// --- Start of Residual Block 3 ---
for (int i=0; i<24576; i++) res_fm_ptr[i] = in_fm[i];
// features.3.conv.0.0 : 32x32x24 -> 32x32x144 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_3_conv_0_0_weights,
    32, 32, 24, 32, 32, 144, 1, 1, 0, 1, features_3_conv_0_0_scale, features_3_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_3_conv_0_0_weights, features_3_conv_0_0_pos, features_3_conv_0_0_mask, features_3_conv_0_0_slots,
    NULL, NULL,
    32, 32, 24, 32, 32, 144, 1, 1, 0, 1, features_3_conv_0_0_d, features_3_conv_0_0_words, features_3_conv_0_0_scale, features_3_conv_0_0_bias);
#endif
apply_relu(out_fm, 147456);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.3.conv.1.0 : 32x32x144 -> 32x32x144 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_3_conv_1_0_weights,
    32, 32, 144, 32, 32, 3, 3, 1, 1, features_3_conv_1_0_scale, features_3_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_3_conv_1_0_weights, features_3_conv_1_0_pos, features_3_conv_1_0_mask, features_3_conv_1_0_slots,
    NULL, NULL,
    32, 32, 144, 32, 32, 3, 3, 1, 1, features_3_conv_1_0_d, features_3_conv_1_0_words, features_3_conv_1_0_scale, features_3_conv_1_0_bias);
#endif
apply_relu(out_fm, 147456);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.3.conv.2 : 32x32x144 -> 32x32x24 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_3_conv_2_weights,
    32, 32, 144, 32, 32, 24, 1, 1, 0, 1, features_3_conv_2_scale, features_3_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_3_conv_2_weights, features_3_conv_2_pos, features_3_conv_2_mask, features_3_conv_2_slots,
    NULL, NULL,
    32, 32, 144, 32, 32, 24, 1, 1, 0, 1, features_3_conv_2_d, features_3_conv_2_words, features_3_conv_2_scale, features_3_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 3 ---
add_tensors(in_fm, res_fm_ptr, 24576);

// features.4.conv.0.0 : 32x32x24 -> 32x32x144 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_4_conv_0_0_weights,
    32, 32, 24, 32, 32, 144, 1, 1, 0, 1, features_4_conv_0_0_scale, features_4_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_4_conv_0_0_weights, features_4_conv_0_0_pos, features_4_conv_0_0_mask, features_4_conv_0_0_slots,
    NULL, NULL,
    32, 32, 24, 32, 32, 144, 1, 1, 0, 1, features_4_conv_0_0_d, features_4_conv_0_0_words, features_4_conv_0_0_scale, features_4_conv_0_0_bias);
#endif
apply_relu(out_fm, 147456);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.4.conv.1.0 : 32x32x144 -> 16x16x144 (s=2, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_4_conv_1_0_weights,
    32, 32, 144, 16, 16, 3, 3, 1, 2, features_4_conv_1_0_scale, features_4_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_4_conv_1_0_weights, features_4_conv_1_0_pos, features_4_conv_1_0_mask, features_4_conv_1_0_slots,
    NULL, NULL,
    32, 32, 144, 16, 16, 3, 3, 1, 2, features_4_conv_1_0_d, features_4_conv_1_0_words, features_4_conv_1_0_scale, features_4_conv_1_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.4.conv.2 : 16x16x144 -> 16x16x32 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_4_conv_2_weights,
    16, 16, 144, 16, 16, 32, 1, 1, 0, 1, features_4_conv_2_scale, features_4_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_4_conv_2_weights, features_4_conv_2_pos, features_4_conv_2_mask, features_4_conv_2_slots,
    NULL, NULL,
    16, 16, 144, 16, 16, 32, 1, 1, 0, 1, features_4_conv_2_d, features_4_conv_2_words, features_4_conv_2_scale, features_4_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// --- Start of Residual Block 5 ---
for (int i=0; i<8192; i++) res_fm_ptr[i] = in_fm[i];
// features.5.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_5_conv_0_0_weights,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_5_conv_0_0_scale, features_5_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_5_conv_0_0_weights, features_5_conv_0_0_pos, features_5_conv_0_0_mask, features_5_conv_0_0_slots,
    NULL, NULL,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_5_conv_0_0_d, features_5_conv_0_0_words, features_5_conv_0_0_scale, features_5_conv_0_0_bias);
#endif
apply_relu(out_fm, 49152);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.5.conv.1.0 : 16x16x192 -> 16x16x192 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_5_conv_1_0_weights,
    16, 16, 192, 16, 16, 3, 3, 1, 1, features_5_conv_1_0_scale, features_5_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_5_conv_1_0_weights, features_5_conv_1_0_pos, features_5_conv_1_0_mask, features_5_conv_1_0_slots,
    NULL, NULL,
    16, 16, 192, 16, 16, 3, 3, 1, 1, features_5_conv_1_0_d, features_5_conv_1_0_words, features_5_conv_1_0_scale, features_5_conv_1_0_bias);
#endif
apply_relu(out_fm, 49152);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.5.conv.2 : 16x16x192 -> 16x16x32 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_5_conv_2_weights,
    16, 16, 192, 16, 16, 32, 1, 1, 0, 1, features_5_conv_2_scale, features_5_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_5_conv_2_weights, features_5_conv_2_pos, features_5_conv_2_mask, features_5_conv_2_slots,
    NULL, NULL,
    16, 16, 192, 16, 16, 32, 1, 1, 0, 1, features_5_conv_2_d, features_5_conv_2_words, features_5_conv_2_scale, features_5_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 5 ---
add_tensors(in_fm, res_fm_ptr, 8192);

// --- Start of Residual Block 6 ---
for (int i=0; i<8192; i++) res_fm_ptr[i] = in_fm[i];
// features.6.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_6_conv_0_0_weights,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_6_conv_0_0_scale, features_6_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_6_conv_0_0_weights, features_6_conv_0_0_pos, features_6_conv_0_0_mask, features_6_conv_0_0_slots,
    NULL, NULL,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_6_conv_0_0_d, features_6_conv_0_0_words, features_6_conv_0_0_scale, features_6_conv_0_0_bias);
#endif
apply_relu(out_fm, 49152);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.6.conv.1.0 : 16x16x192 -> 16x16x192 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_6_conv_1_0_weights,
    16, 16, 192, 16, 16, 3, 3, 1, 1, features_6_conv_1_0_scale, features_6_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_6_conv_1_0_weights, features_6_conv_1_0_pos, features_6_conv_1_0_mask, features_6_conv_1_0_slots,
    NULL, NULL,
    16, 16, 192, 16, 16, 3, 3, 1, 1, features_6_conv_1_0_d, features_6_conv_1_0_words, features_6_conv_1_0_scale, features_6_conv_1_0_bias);
#endif
apply_relu(out_fm, 49152);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.6.conv.2 : 16x16x192 -> 16x16x32 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_6_conv_2_weights,
    16, 16, 192, 16, 16, 32, 1, 1, 0, 1, features_6_conv_2_scale, features_6_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_6_conv_2_weights, features_6_conv_2_pos, features_6_conv_2_mask, features_6_conv_2_slots,
    NULL, NULL,
    16, 16, 192, 16, 16, 32, 1, 1, 0, 1, features_6_conv_2_d, features_6_conv_2_words, features_6_conv_2_scale, features_6_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 6 ---
add_tensors(in_fm, res_fm_ptr, 8192);

// features.7.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_7_conv_0_0_weights,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_7_conv_0_0_scale, features_7_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_7_conv_0_0_weights, features_7_conv_0_0_pos, features_7_conv_0_0_mask, features_7_conv_0_0_slots,
    NULL, NULL,
    16, 16, 32, 16, 16, 192, 1, 1, 0, 1, features_7_conv_0_0_d, features_7_conv_0_0_words, features_7_conv_0_0_scale, features_7_conv_0_0_bias);
#endif
apply_relu(out_fm, 49152);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.7.conv.1.0 : 16x16x192 -> 8x8x192 (s=2, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_7_conv_1_0_weights,
    16, 16, 192, 8, 8, 3, 3, 1, 2, features_7_conv_1_0_scale, features_7_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_7_conv_1_0_weights, features_7_conv_1_0_pos, features_7_conv_1_0_mask, features_7_conv_1_0_slots,
    NULL, NULL,
    16, 16, 192, 8, 8, 3, 3, 1, 2, features_7_conv_1_0_d, features_7_conv_1_0_words, features_7_conv_1_0_scale, features_7_conv_1_0_bias);
#endif
apply_relu(out_fm, 12288);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.7.conv.2 : 8x8x192 -> 8x8x64 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_7_conv_2_weights,
    8, 8, 192, 8, 8, 64, 1, 1, 0, 1, features_7_conv_2_scale, features_7_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_7_conv_2_weights, features_7_conv_2_pos, features_7_conv_2_mask, features_7_conv_2_slots,
    NULL, NULL,
    8, 8, 192, 8, 8, 64, 1, 1, 0, 1, features_7_conv_2_d, features_7_conv_2_words, features_7_conv_2_scale, features_7_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// --- Start of Residual Block 8 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = in_fm[i];
// features.8.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_8_conv_0_0_weights,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_8_conv_0_0_scale, features_8_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_8_conv_0_0_weights, features_8_conv_0_0_pos, features_8_conv_0_0_mask, features_8_conv_0_0_slots,
    NULL, NULL,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_8_conv_0_0_d, features_8_conv_0_0_words, features_8_conv_0_0_scale, features_8_conv_0_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.8.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_8_conv_1_0_weights,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_8_conv_1_0_scale, features_8_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_8_conv_1_0_weights, features_8_conv_1_0_pos, features_8_conv_1_0_mask, features_8_conv_1_0_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_8_conv_1_0_d, features_8_conv_1_0_words, features_8_conv_1_0_scale, features_8_conv_1_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.8.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_8_conv_2_weights,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_8_conv_2_scale, features_8_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_8_conv_2_weights, features_8_conv_2_pos, features_8_conv_2_mask, features_8_conv_2_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_8_conv_2_d, features_8_conv_2_words, features_8_conv_2_scale, features_8_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 8 ---
add_tensors(in_fm, res_fm_ptr, 4096);

// --- Start of Residual Block 9 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = in_fm[i];
// features.9.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_9_conv_0_0_weights,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_9_conv_0_0_scale, features_9_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_9_conv_0_0_weights, features_9_conv_0_0_pos, features_9_conv_0_0_mask, features_9_conv_0_0_slots,
    NULL, NULL,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_9_conv_0_0_d, features_9_conv_0_0_words, features_9_conv_0_0_scale, features_9_conv_0_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.9.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_9_conv_1_0_weights,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_9_conv_1_0_scale, features_9_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_9_conv_1_0_weights, features_9_conv_1_0_pos, features_9_conv_1_0_mask, features_9_conv_1_0_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_9_conv_1_0_d, features_9_conv_1_0_words, features_9_conv_1_0_scale, features_9_conv_1_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.9.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_9_conv_2_weights,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_9_conv_2_scale, features_9_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_9_conv_2_weights, features_9_conv_2_pos, features_9_conv_2_mask, features_9_conv_2_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_9_conv_2_d, features_9_conv_2_words, features_9_conv_2_scale, features_9_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 9 ---
add_tensors(in_fm, res_fm_ptr, 4096);

// --- Start of Residual Block 10 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = in_fm[i];
// features.10.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_10_conv_0_0_weights,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_10_conv_0_0_scale, features_10_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_10_conv_0_0_weights, features_10_conv_0_0_pos, features_10_conv_0_0_mask, features_10_conv_0_0_slots,
    NULL, NULL,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_10_conv_0_0_d, features_10_conv_0_0_words, features_10_conv_0_0_scale, features_10_conv_0_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.10.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_10_conv_1_0_weights,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_10_conv_1_0_scale, features_10_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_10_conv_1_0_weights, features_10_conv_1_0_pos, features_10_conv_1_0_mask, features_10_conv_1_0_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_10_conv_1_0_d, features_10_conv_1_0_words, features_10_conv_1_0_scale, features_10_conv_1_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.10.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_10_conv_2_weights,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_10_conv_2_scale, features_10_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_10_conv_2_weights, features_10_conv_2_pos, features_10_conv_2_mask, features_10_conv_2_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 64, 1, 1, 0, 1, features_10_conv_2_d, features_10_conv_2_words, features_10_conv_2_scale, features_10_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 10 ---
add_tensors(in_fm, res_fm_ptr, 4096);

// features.11.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_11_conv_0_0_weights,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_11_conv_0_0_scale, features_11_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_11_conv_0_0_weights, features_11_conv_0_0_pos, features_11_conv_0_0_mask, features_11_conv_0_0_slots,
    NULL, NULL,
    8, 8, 64, 8, 8, 384, 1, 1, 0, 1, features_11_conv_0_0_d, features_11_conv_0_0_words, features_11_conv_0_0_scale, features_11_conv_0_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.11.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_11_conv_1_0_weights,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_11_conv_1_0_scale, features_11_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_11_conv_1_0_weights, features_11_conv_1_0_pos, features_11_conv_1_0_mask, features_11_conv_1_0_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 3, 3, 1, 1, features_11_conv_1_0_d, features_11_conv_1_0_words, features_11_conv_1_0_scale, features_11_conv_1_0_bias);
#endif
apply_relu(out_fm, 24576);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.11.conv.2 : 8x8x384 -> 8x8x96 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_11_conv_2_weights,
    8, 8, 384, 8, 8, 96, 1, 1, 0, 1, features_11_conv_2_scale, features_11_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_11_conv_2_weights, features_11_conv_2_pos, features_11_conv_2_mask, features_11_conv_2_slots,
    NULL, NULL,
    8, 8, 384, 8, 8, 96, 1, 1, 0, 1, features_11_conv_2_d, features_11_conv_2_words, features_11_conv_2_scale, features_11_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// --- Start of Residual Block 12 ---
for (int i=0; i<6144; i++) res_fm_ptr[i] = in_fm[i];
// features.12.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_12_conv_0_0_weights,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_12_conv_0_0_scale, features_12_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_12_conv_0_0_weights, features_12_conv_0_0_pos, features_12_conv_0_0_mask, features_12_conv_0_0_slots,
    NULL, NULL,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_12_conv_0_0_d, features_12_conv_0_0_words, features_12_conv_0_0_scale, features_12_conv_0_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.12.conv.1.0 : 8x8x576 -> 8x8x576 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_12_conv_1_0_weights,
    8, 8, 576, 8, 8, 3, 3, 1, 1, features_12_conv_1_0_scale, features_12_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_12_conv_1_0_weights, features_12_conv_1_0_pos, features_12_conv_1_0_mask, features_12_conv_1_0_slots,
    NULL, NULL,
    8, 8, 576, 8, 8, 3, 3, 1, 1, features_12_conv_1_0_d, features_12_conv_1_0_words, features_12_conv_1_0_scale, features_12_conv_1_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.12.conv.2 : 8x8x576 -> 8x8x96 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_12_conv_2_weights,
    8, 8, 576, 8, 8, 96, 1, 1, 0, 1, features_12_conv_2_scale, features_12_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_12_conv_2_weights, features_12_conv_2_pos, features_12_conv_2_mask, features_12_conv_2_slots,
    NULL, NULL,
    8, 8, 576, 8, 8, 96, 1, 1, 0, 1, features_12_conv_2_d, features_12_conv_2_words, features_12_conv_2_scale, features_12_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 12 ---
add_tensors(in_fm, res_fm_ptr, 6144);

// --- Start of Residual Block 13 ---
for (int i=0; i<6144; i++) res_fm_ptr[i] = in_fm[i];
// features.13.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_13_conv_0_0_weights,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_13_conv_0_0_scale, features_13_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_13_conv_0_0_weights, features_13_conv_0_0_pos, features_13_conv_0_0_mask, features_13_conv_0_0_slots,
    NULL, NULL,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_13_conv_0_0_d, features_13_conv_0_0_words, features_13_conv_0_0_scale, features_13_conv_0_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.13.conv.1.0 : 8x8x576 -> 8x8x576 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_13_conv_1_0_weights,
    8, 8, 576, 8, 8, 3, 3, 1, 1, features_13_conv_1_0_scale, features_13_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_13_conv_1_0_weights, features_13_conv_1_0_pos, features_13_conv_1_0_mask, features_13_conv_1_0_slots,
    NULL, NULL,
    8, 8, 576, 8, 8, 3, 3, 1, 1, features_13_conv_1_0_d, features_13_conv_1_0_words, features_13_conv_1_0_scale, features_13_conv_1_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.13.conv.2 : 8x8x576 -> 8x8x96 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_13_conv_2_weights,
    8, 8, 576, 8, 8, 96, 1, 1, 0, 1, features_13_conv_2_scale, features_13_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_13_conv_2_weights, features_13_conv_2_pos, features_13_conv_2_mask, features_13_conv_2_slots,
    NULL, NULL,
    8, 8, 576, 8, 8, 96, 1, 1, 0, 1, features_13_conv_2_d, features_13_conv_2_words, features_13_conv_2_scale, features_13_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 13 ---
add_tensors(in_fm, res_fm_ptr, 6144);

// features.14.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_14_conv_0_0_weights,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_14_conv_0_0_scale, features_14_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_14_conv_0_0_weights, features_14_conv_0_0_pos, features_14_conv_0_0_mask, features_14_conv_0_0_slots,
    NULL, NULL,
    8, 8, 96, 8, 8, 576, 1, 1, 0, 1, features_14_conv_0_0_d, features_14_conv_0_0_words, features_14_conv_0_0_scale, features_14_conv_0_0_bias);
#endif
apply_relu(out_fm, 36864);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.14.conv.1.0 : 8x8x576 -> 4x4x576 (s=2, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_14_conv_1_0_weights,
    8, 8, 576, 4, 4, 3, 3, 1, 2, features_14_conv_1_0_scale, features_14_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_14_conv_1_0_weights, features_14_conv_1_0_pos, features_14_conv_1_0_mask, features_14_conv_1_0_slots,
    NULL, NULL,
    8, 8, 576, 4, 4, 3, 3, 1, 2, features_14_conv_1_0_d, features_14_conv_1_0_words, features_14_conv_1_0_scale, features_14_conv_1_0_bias);
#endif
apply_relu(out_fm, 9216);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.14.conv.2 : 4x4x576 -> 4x4x160 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_14_conv_2_weights,
    4, 4, 576, 4, 4, 160, 1, 1, 0, 1, features_14_conv_2_scale, features_14_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_14_conv_2_weights, features_14_conv_2_pos, features_14_conv_2_mask, features_14_conv_2_slots,
    NULL, NULL,
    4, 4, 576, 4, 4, 160, 1, 1, 0, 1, features_14_conv_2_d, features_14_conv_2_words, features_14_conv_2_scale, features_14_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// --- Start of Residual Block 15 ---
for (int i=0; i<2560; i++) res_fm_ptr[i] = in_fm[i];
// features.15.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_15_conv_0_0_weights,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_15_conv_0_0_scale, features_15_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_15_conv_0_0_weights, features_15_conv_0_0_pos, features_15_conv_0_0_mask, features_15_conv_0_0_slots,
    NULL, NULL,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_15_conv_0_0_d, features_15_conv_0_0_words, features_15_conv_0_0_scale, features_15_conv_0_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.15.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_15_conv_1_0_weights,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_15_conv_1_0_scale, features_15_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_15_conv_1_0_weights, features_15_conv_1_0_pos, features_15_conv_1_0_mask, features_15_conv_1_0_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_15_conv_1_0_d, features_15_conv_1_0_words, features_15_conv_1_0_scale, features_15_conv_1_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.15.conv.2 : 4x4x960 -> 4x4x160 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_15_conv_2_weights,
    4, 4, 960, 4, 4, 160, 1, 1, 0, 1, features_15_conv_2_scale, features_15_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_15_conv_2_weights, features_15_conv_2_pos, features_15_conv_2_mask, features_15_conv_2_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 160, 1, 1, 0, 1, features_15_conv_2_d, features_15_conv_2_words, features_15_conv_2_scale, features_15_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 15 ---
add_tensors(in_fm, res_fm_ptr, 2560);

// --- Start of Residual Block 16 ---
for (int i=0; i<2560; i++) res_fm_ptr[i] = in_fm[i];
// features.16.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_16_conv_0_0_weights,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_16_conv_0_0_scale, features_16_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_16_conv_0_0_weights, features_16_conv_0_0_pos, features_16_conv_0_0_mask, features_16_conv_0_0_slots,
    NULL, NULL,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_16_conv_0_0_d, features_16_conv_0_0_words, features_16_conv_0_0_scale, features_16_conv_0_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.16.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_16_conv_1_0_weights,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_16_conv_1_0_scale, features_16_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_16_conv_1_0_weights, features_16_conv_1_0_pos, features_16_conv_1_0_mask, features_16_conv_1_0_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_16_conv_1_0_d, features_16_conv_1_0_words, features_16_conv_1_0_scale, features_16_conv_1_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.16.conv.2 : 4x4x960 -> 4x4x160 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_16_conv_2_weights,
    4, 4, 960, 4, 4, 160, 1, 1, 0, 1, features_16_conv_2_scale, features_16_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_16_conv_2_weights, features_16_conv_2_pos, features_16_conv_2_mask, features_16_conv_2_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 160, 1, 1, 0, 1, features_16_conv_2_d, features_16_conv_2_words, features_16_conv_2_scale, features_16_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
// --- End of Residual Block 16 ---
add_tensors(in_fm, res_fm_ptr, 2560);

// features.17.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_17_conv_0_0_weights,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_17_conv_0_0_scale, features_17_conv_0_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_17_conv_0_0_weights, features_17_conv_0_0_pos, features_17_conv_0_0_mask, features_17_conv_0_0_slots,
    NULL, NULL,
    4, 4, 160, 4, 4, 960, 1, 1, 0, 1, features_17_conv_0_0_d, features_17_conv_0_0_words, features_17_conv_0_0_scale, features_17_conv_0_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.17.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
#ifdef BASELINE_MODE
run_depthwise_conv2d_8bit(in_fm, out_fm, features_17_conv_1_0_weights,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_17_conv_1_0_scale, features_17_conv_1_0_bias);
#else
run_depthwise_conv2d_general(in_fm, out_fm, features_17_conv_1_0_weights, features_17_conv_1_0_pos, features_17_conv_1_0_mask, features_17_conv_1_0_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 3, 3, 1, 1, features_17_conv_1_0_d, features_17_conv_1_0_words, features_17_conv_1_0_scale, features_17_conv_1_0_bias);
#endif
apply_relu(out_fm, 15360);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.17.conv.2 : 4x4x960 -> 4x4x320 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_17_conv_2_weights,
    4, 4, 960, 4, 4, 320, 1, 1, 0, 1, features_17_conv_2_scale, features_17_conv_2_bias);
#else
run_conv2d_general(in_fm, out_fm, features_17_conv_2_weights, features_17_conv_2_pos, features_17_conv_2_mask, features_17_conv_2_slots,
    NULL, NULL,
    4, 4, 960, 4, 4, 320, 1, 1, 0, 1, features_17_conv_2_d, features_17_conv_2_words, features_17_conv_2_scale, features_17_conv_2_bias);
#endif
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// features.18.0 : 4x4x320 -> 4x4x1280 (s=1, p=0)
#ifdef BASELINE_MODE
run_conv2d_8bit(in_fm, out_fm, features_18_0_weights,
    4, 4, 320, 4, 4, 1280, 1, 1, 0, 1, features_18_0_scale, features_18_0_bias);
#else
run_conv2d_general(in_fm, out_fm, features_18_0_weights, features_18_0_pos, features_18_0_mask, features_18_0_slots,
    NULL, NULL,
    4, 4, 320, 4, 4, 1280, 1, 1, 0, 1, features_18_0_d, features_18_0_words, features_18_0_scale, features_18_0_bias);
#endif
apply_relu(out_fm, 20480);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");

// classifier.1
global_average_pool_2d(in_fm, out_fm, 4, 4, 1280);
tmp = in_fm; in_fm = out_fm; out_fm = tmp;
printf("features.0.0 first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
#ifdef BASELINE_MODE
run_linear_8bit(in_fm, out_fm, classifier_1_weights,
    1280, 10, classifier_1_scale, classifier_1_bias);
#else
run_linear_general(in_fm, out_fm, classifier_1_weights, classifier_1_pos, classifier_1_mask, classifier_1_slots,
    NULL, NULL,
    1280, 10, classifier_1_d, classifier_1_words, classifier_1_scale, classifier_1_bias);
#endif
in_fm = out_fm;


    printf("Classifier Input first 10: "); for(int _i=0; _i<10; _i++) printf("%d ", in_fm[_i]); printf("\\n");
printf("\\n=== Classification Results ===\\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\n", i, out_fm[i]);
    }
    printf("=== End Forward Pass ===\n\n");
    print_mobilenet_profiler_summary(mode_name);
}
#else
const uint8_t dummy_8_act_bits[3] = {8, 8, 8};
static int run_full_mobilenet_packed_forward_pass(void) {
    const char *mode_name = "MobileNet Packed MQF";
    printf("\n=== Starting Full %s Forward Pass ===\n", mode_name);
    printf("Input Image: CIFAR-10 (3x32x32)\n");
    total_memory_loads = 0;

/* Auto-generated MobileNet Packed Pipeline */
int8_t *unpacked_fm = fm_B;
int8_t *packed_fm = fm_A;
int8_t *res_fm_ptr = res_fm;
uint32_t current_offsets[4096];
const uint8_t *prev_act_bits;

// Initial state: unpacked_fm has the image.
for (int i=0; i<3072; i++) unpacked_fm[i] = current_test_image[i];
prev_act_bits = dummy_8_act_bits;

// features.0.0 : 32x32x3 -> 32x32x32 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 3, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 3, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_0_0_weights,
    features_0_0_pos, features_0_0_mask, features_0_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 3,
    32, 32, 32,
    3, 3, 1, 1,
    features_0_0_scale, features_0_0_bias,
    FEATURES_0_0_MAX_D, FEATURES_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 32768);
prev_act_bits = features_0_0_out_act_bits;

// features.1.conv.0.0 : 32x32x32 -> 32x32x32 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 32, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 32, 32 * 32);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_1_conv_0_0_weights,
    features_1_conv_0_0_pos, features_1_conv_0_0_mask, features_1_conv_0_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 32,
    32, 32,
    3, 3, 1, 1,
    features_1_conv_0_0_scale, features_1_conv_0_0_bias,
    FEATURES_1_CONV_0_0_MAX_D, FEATURES_1_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 32768);
prev_act_bits = features_1_conv_0_0_out_act_bits;

// features.1.conv.1 : 32x32x32 -> 32x32x16 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 32, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 32, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_1_conv_1_weights,
    features_1_conv_1_pos, features_1_conv_1_mask, features_1_conv_1_slots,
    prev_act_bits, current_offsets,
    32, 32, 32,
    32, 32, 16,
    1, 1, 0, 1,
    features_1_conv_1_scale, features_1_conv_1_bias,
    FEATURES_1_CONV_1_MAX_D, FEATURES_1_CONV_1_WORDS_PER_FILTER);
prev_act_bits = features_1_conv_1_out_act_bits;

// features.2.conv.0.0 : 32x32x16 -> 32x32x96 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 16, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 16, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_2_conv_0_0_weights,
    features_2_conv_0_0_pos, features_2_conv_0_0_mask, features_2_conv_0_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 16,
    32, 32, 96,
    1, 1, 0, 1,
    features_2_conv_0_0_scale, features_2_conv_0_0_bias,
    FEATURES_2_CONV_0_0_MAX_D, FEATURES_2_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 98304);
prev_act_bits = features_2_conv_0_0_out_act_bits;

// features.2.conv.1.0 : 32x32x96 -> 32x32x96 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 96, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 96, 32 * 32);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_2_conv_1_0_weights,
    features_2_conv_1_0_pos, features_2_conv_1_0_mask, features_2_conv_1_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 96,
    32, 32,
    3, 3, 1, 1,
    features_2_conv_1_0_scale, features_2_conv_1_0_bias,
    FEATURES_2_CONV_1_0_MAX_D, FEATURES_2_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 98304);
prev_act_bits = features_2_conv_1_0_out_act_bits;

// features.2.conv.2 : 32x32x96 -> 32x32x24 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 96, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 96, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_2_conv_2_weights,
    features_2_conv_2_pos, features_2_conv_2_mask, features_2_conv_2_slots,
    prev_act_bits, current_offsets,
    32, 32, 96,
    32, 32, 24,
    1, 1, 0, 1,
    features_2_conv_2_scale, features_2_conv_2_bias,
    FEATURES_2_CONV_2_MAX_D, FEATURES_2_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_2_conv_2_out_act_bits;

// --- Start of Residual Block 3 ---
for (int i=0; i<24576; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.3.conv.0.0 : 32x32x24 -> 32x32x144 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 24, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 24, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_3_conv_0_0_weights,
    features_3_conv_0_0_pos, features_3_conv_0_0_mask, features_3_conv_0_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 24,
    32, 32, 144,
    1, 1, 0, 1,
    features_3_conv_0_0_scale, features_3_conv_0_0_bias,
    FEATURES_3_CONV_0_0_MAX_D, FEATURES_3_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 147456);
prev_act_bits = features_3_conv_0_0_out_act_bits;

// features.3.conv.1.0 : 32x32x144 -> 32x32x144 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 144, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 144, 32 * 32);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_3_conv_1_0_weights,
    features_3_conv_1_0_pos, features_3_conv_1_0_mask, features_3_conv_1_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 144,
    32, 32,
    3, 3, 1, 1,
    features_3_conv_1_0_scale, features_3_conv_1_0_bias,
    FEATURES_3_CONV_1_0_MAX_D, FEATURES_3_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 147456);
prev_act_bits = features_3_conv_1_0_out_act_bits;

// features.3.conv.2 : 32x32x144 -> 32x32x24 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 144, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 144, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_3_conv_2_weights,
    features_3_conv_2_pos, features_3_conv_2_mask, features_3_conv_2_slots,
    prev_act_bits, current_offsets,
    32, 32, 144,
    32, 32, 24,
    1, 1, 0, 1,
    features_3_conv_2_scale, features_3_conv_2_bias,
    FEATURES_3_CONV_2_MAX_D, FEATURES_3_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_3_conv_2_out_act_bits;
// --- End of Residual Block 3 ---
add_tensors(unpacked_fm, res_fm_ptr, 24576);

// features.4.conv.0.0 : 32x32x24 -> 32x32x144 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 24, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 24, 32 * 32);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_4_conv_0_0_weights,
    features_4_conv_0_0_pos, features_4_conv_0_0_mask, features_4_conv_0_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 24,
    32, 32, 144,
    1, 1, 0, 1,
    features_4_conv_0_0_scale, features_4_conv_0_0_bias,
    FEATURES_4_CONV_0_0_MAX_D, FEATURES_4_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 147456);
prev_act_bits = features_4_conv_0_0_out_act_bits;

// features.4.conv.1.0 : 32x32x144 -> 16x16x144 (s=2, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 144, 32 * 32);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 144, 32 * 32);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_4_conv_1_0_weights,
    features_4_conv_1_0_pos, features_4_conv_1_0_mask, features_4_conv_1_0_slots,
    prev_act_bits, current_offsets,
    32, 32, 144,
    16, 16,
    3, 3, 1, 2,
    features_4_conv_1_0_scale, features_4_conv_1_0_bias,
    FEATURES_4_CONV_1_0_MAX_D, FEATURES_4_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_4_conv_1_0_out_act_bits;

// features.4.conv.2 : 16x16x144 -> 16x16x32 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 144, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 144, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_4_conv_2_weights,
    features_4_conv_2_pos, features_4_conv_2_mask, features_4_conv_2_slots,
    prev_act_bits, current_offsets,
    16, 16, 144,
    16, 16, 32,
    1, 1, 0, 1,
    features_4_conv_2_scale, features_4_conv_2_bias,
    FEATURES_4_CONV_2_MAX_D, FEATURES_4_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_4_conv_2_out_act_bits;

// --- Start of Residual Block 5 ---
for (int i=0; i<8192; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.5.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 32, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 32, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_5_conv_0_0_weights,
    features_5_conv_0_0_pos, features_5_conv_0_0_mask, features_5_conv_0_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 32,
    16, 16, 192,
    1, 1, 0, 1,
    features_5_conv_0_0_scale, features_5_conv_0_0_bias,
    FEATURES_5_CONV_0_0_MAX_D, FEATURES_5_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 49152);
prev_act_bits = features_5_conv_0_0_out_act_bits;

// features.5.conv.1.0 : 16x16x192 -> 16x16x192 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 192, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 16 * 16);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_5_conv_1_0_weights,
    features_5_conv_1_0_pos, features_5_conv_1_0_mask, features_5_conv_1_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 192,
    16, 16,
    3, 3, 1, 1,
    features_5_conv_1_0_scale, features_5_conv_1_0_bias,
    FEATURES_5_CONV_1_0_MAX_D, FEATURES_5_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 49152);
prev_act_bits = features_5_conv_1_0_out_act_bits;

// features.5.conv.2 : 16x16x192 -> 16x16x32 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 192, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_5_conv_2_weights,
    features_5_conv_2_pos, features_5_conv_2_mask, features_5_conv_2_slots,
    prev_act_bits, current_offsets,
    16, 16, 192,
    16, 16, 32,
    1, 1, 0, 1,
    features_5_conv_2_scale, features_5_conv_2_bias,
    FEATURES_5_CONV_2_MAX_D, FEATURES_5_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_5_conv_2_out_act_bits;
// --- End of Residual Block 5 ---
add_tensors(unpacked_fm, res_fm_ptr, 8192);

// --- Start of Residual Block 6 ---
for (int i=0; i<8192; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.6.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 32, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 32, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_6_conv_0_0_weights,
    features_6_conv_0_0_pos, features_6_conv_0_0_mask, features_6_conv_0_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 32,
    16, 16, 192,
    1, 1, 0, 1,
    features_6_conv_0_0_scale, features_6_conv_0_0_bias,
    FEATURES_6_CONV_0_0_MAX_D, FEATURES_6_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 49152);
prev_act_bits = features_6_conv_0_0_out_act_bits;

// features.6.conv.1.0 : 16x16x192 -> 16x16x192 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 192, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 16 * 16);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_6_conv_1_0_weights,
    features_6_conv_1_0_pos, features_6_conv_1_0_mask, features_6_conv_1_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 192,
    16, 16,
    3, 3, 1, 1,
    features_6_conv_1_0_scale, features_6_conv_1_0_bias,
    FEATURES_6_CONV_1_0_MAX_D, FEATURES_6_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 49152);
prev_act_bits = features_6_conv_1_0_out_act_bits;

// features.6.conv.2 : 16x16x192 -> 16x16x32 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 192, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_6_conv_2_weights,
    features_6_conv_2_pos, features_6_conv_2_mask, features_6_conv_2_slots,
    prev_act_bits, current_offsets,
    16, 16, 192,
    16, 16, 32,
    1, 1, 0, 1,
    features_6_conv_2_scale, features_6_conv_2_bias,
    FEATURES_6_CONV_2_MAX_D, FEATURES_6_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_6_conv_2_out_act_bits;
// --- End of Residual Block 6 ---
add_tensors(unpacked_fm, res_fm_ptr, 8192);

// features.7.conv.0.0 : 16x16x32 -> 16x16x192 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 32, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 32, 16 * 16);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_7_conv_0_0_weights,
    features_7_conv_0_0_pos, features_7_conv_0_0_mask, features_7_conv_0_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 32,
    16, 16, 192,
    1, 1, 0, 1,
    features_7_conv_0_0_scale, features_7_conv_0_0_bias,
    FEATURES_7_CONV_0_0_MAX_D, FEATURES_7_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 49152);
prev_act_bits = features_7_conv_0_0_out_act_bits;

// features.7.conv.1.0 : 16x16x192 -> 8x8x192 (s=2, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 192, 16 * 16);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 16 * 16);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_7_conv_1_0_weights,
    features_7_conv_1_0_pos, features_7_conv_1_0_mask, features_7_conv_1_0_slots,
    prev_act_bits, current_offsets,
    16, 16, 192,
    8, 8,
    3, 3, 1, 2,
    features_7_conv_1_0_scale, features_7_conv_1_0_bias,
    FEATURES_7_CONV_1_0_MAX_D, FEATURES_7_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 12288);
prev_act_bits = features_7_conv_1_0_out_act_bits;

// features.7.conv.2 : 8x8x192 -> 8x8x64 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 192, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 192, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_7_conv_2_weights,
    features_7_conv_2_pos, features_7_conv_2_mask, features_7_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 192,
    8, 8, 64,
    1, 1, 0, 1,
    features_7_conv_2_scale, features_7_conv_2_bias,
    FEATURES_7_CONV_2_MAX_D, FEATURES_7_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_7_conv_2_out_act_bits;

// --- Start of Residual Block 8 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.8.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 64, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 64, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_8_conv_0_0_weights,
    features_8_conv_0_0_pos, features_8_conv_0_0_mask, features_8_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 64,
    8, 8, 384,
    1, 1, 0, 1,
    features_8_conv_0_0_scale, features_8_conv_0_0_bias,
    FEATURES_8_CONV_0_0_MAX_D, FEATURES_8_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_8_conv_0_0_out_act_bits;

// features.8.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_8_conv_1_0_weights,
    features_8_conv_1_0_pos, features_8_conv_1_0_mask, features_8_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8,
    3, 3, 1, 1,
    features_8_conv_1_0_scale, features_8_conv_1_0_bias,
    FEATURES_8_CONV_1_0_MAX_D, FEATURES_8_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_8_conv_1_0_out_act_bits;

// features.8.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_8_conv_2_weights,
    features_8_conv_2_pos, features_8_conv_2_mask, features_8_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8, 64,
    1, 1, 0, 1,
    features_8_conv_2_scale, features_8_conv_2_bias,
    FEATURES_8_CONV_2_MAX_D, FEATURES_8_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_8_conv_2_out_act_bits;
// --- End of Residual Block 8 ---
add_tensors(unpacked_fm, res_fm_ptr, 4096);

// --- Start of Residual Block 9 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.9.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 64, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 64, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_9_conv_0_0_weights,
    features_9_conv_0_0_pos, features_9_conv_0_0_mask, features_9_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 64,
    8, 8, 384,
    1, 1, 0, 1,
    features_9_conv_0_0_scale, features_9_conv_0_0_bias,
    FEATURES_9_CONV_0_0_MAX_D, FEATURES_9_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_9_conv_0_0_out_act_bits;

// features.9.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_9_conv_1_0_weights,
    features_9_conv_1_0_pos, features_9_conv_1_0_mask, features_9_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8,
    3, 3, 1, 1,
    features_9_conv_1_0_scale, features_9_conv_1_0_bias,
    FEATURES_9_CONV_1_0_MAX_D, FEATURES_9_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_9_conv_1_0_out_act_bits;

// features.9.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_9_conv_2_weights,
    features_9_conv_2_pos, features_9_conv_2_mask, features_9_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8, 64,
    1, 1, 0, 1,
    features_9_conv_2_scale, features_9_conv_2_bias,
    FEATURES_9_CONV_2_MAX_D, FEATURES_9_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_9_conv_2_out_act_bits;
// --- End of Residual Block 9 ---
add_tensors(unpacked_fm, res_fm_ptr, 4096);

// --- Start of Residual Block 10 ---
for (int i=0; i<4096; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.10.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 64, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 64, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_10_conv_0_0_weights,
    features_10_conv_0_0_pos, features_10_conv_0_0_mask, features_10_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 64,
    8, 8, 384,
    1, 1, 0, 1,
    features_10_conv_0_0_scale, features_10_conv_0_0_bias,
    FEATURES_10_CONV_0_0_MAX_D, FEATURES_10_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_10_conv_0_0_out_act_bits;

// features.10.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_10_conv_1_0_weights,
    features_10_conv_1_0_pos, features_10_conv_1_0_mask, features_10_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8,
    3, 3, 1, 1,
    features_10_conv_1_0_scale, features_10_conv_1_0_bias,
    FEATURES_10_CONV_1_0_MAX_D, FEATURES_10_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_10_conv_1_0_out_act_bits;

// features.10.conv.2 : 8x8x384 -> 8x8x64 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_10_conv_2_weights,
    features_10_conv_2_pos, features_10_conv_2_mask, features_10_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8, 64,
    1, 1, 0, 1,
    features_10_conv_2_scale, features_10_conv_2_bias,
    FEATURES_10_CONV_2_MAX_D, FEATURES_10_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_10_conv_2_out_act_bits;
// --- End of Residual Block 10 ---
add_tensors(unpacked_fm, res_fm_ptr, 4096);

// features.11.conv.0.0 : 8x8x64 -> 8x8x384 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 64, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 64, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_11_conv_0_0_weights,
    features_11_conv_0_0_pos, features_11_conv_0_0_mask, features_11_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 64,
    8, 8, 384,
    1, 1, 0, 1,
    features_11_conv_0_0_scale, features_11_conv_0_0_bias,
    FEATURES_11_CONV_0_0_MAX_D, FEATURES_11_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_11_conv_0_0_out_act_bits;

// features.11.conv.1.0 : 8x8x384 -> 8x8x384 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_11_conv_1_0_weights,
    features_11_conv_1_0_pos, features_11_conv_1_0_mask, features_11_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8,
    3, 3, 1, 1,
    features_11_conv_1_0_scale, features_11_conv_1_0_bias,
    FEATURES_11_CONV_1_0_MAX_D, FEATURES_11_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 24576);
prev_act_bits = features_11_conv_1_0_out_act_bits;

// features.11.conv.2 : 8x8x384 -> 8x8x96 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 384, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 384, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_11_conv_2_weights,
    features_11_conv_2_pos, features_11_conv_2_mask, features_11_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 384,
    8, 8, 96,
    1, 1, 0, 1,
    features_11_conv_2_scale, features_11_conv_2_bias,
    FEATURES_11_CONV_2_MAX_D, FEATURES_11_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_11_conv_2_out_act_bits;

// --- Start of Residual Block 12 ---
for (int i=0; i<6144; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.12.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 96, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 96, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_12_conv_0_0_weights,
    features_12_conv_0_0_pos, features_12_conv_0_0_mask, features_12_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 96,
    8, 8, 576,
    1, 1, 0, 1,
    features_12_conv_0_0_scale, features_12_conv_0_0_bias,
    FEATURES_12_CONV_0_0_MAX_D, FEATURES_12_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_12_conv_0_0_out_act_bits;

// features.12.conv.1.0 : 8x8x576 -> 8x8x576 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 576, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_12_conv_1_0_weights,
    features_12_conv_1_0_pos, features_12_conv_1_0_mask, features_12_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 576,
    8, 8,
    3, 3, 1, 1,
    features_12_conv_1_0_scale, features_12_conv_1_0_bias,
    FEATURES_12_CONV_1_0_MAX_D, FEATURES_12_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_12_conv_1_0_out_act_bits;

// features.12.conv.2 : 8x8x576 -> 8x8x96 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 576, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_12_conv_2_weights,
    features_12_conv_2_pos, features_12_conv_2_mask, features_12_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 576,
    8, 8, 96,
    1, 1, 0, 1,
    features_12_conv_2_scale, features_12_conv_2_bias,
    FEATURES_12_CONV_2_MAX_D, FEATURES_12_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_12_conv_2_out_act_bits;
// --- End of Residual Block 12 ---
add_tensors(unpacked_fm, res_fm_ptr, 6144);

// --- Start of Residual Block 13 ---
for (int i=0; i<6144; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.13.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 96, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 96, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_13_conv_0_0_weights,
    features_13_conv_0_0_pos, features_13_conv_0_0_mask, features_13_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 96,
    8, 8, 576,
    1, 1, 0, 1,
    features_13_conv_0_0_scale, features_13_conv_0_0_bias,
    FEATURES_13_CONV_0_0_MAX_D, FEATURES_13_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_13_conv_0_0_out_act_bits;

// features.13.conv.1.0 : 8x8x576 -> 8x8x576 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 576, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_13_conv_1_0_weights,
    features_13_conv_1_0_pos, features_13_conv_1_0_mask, features_13_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 576,
    8, 8,
    3, 3, 1, 1,
    features_13_conv_1_0_scale, features_13_conv_1_0_bias,
    FEATURES_13_CONV_1_0_MAX_D, FEATURES_13_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_13_conv_1_0_out_act_bits;

// features.13.conv.2 : 8x8x576 -> 8x8x96 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 576, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_13_conv_2_weights,
    features_13_conv_2_pos, features_13_conv_2_mask, features_13_conv_2_slots,
    prev_act_bits, current_offsets,
    8, 8, 576,
    8, 8, 96,
    1, 1, 0, 1,
    features_13_conv_2_scale, features_13_conv_2_bias,
    FEATURES_13_CONV_2_MAX_D, FEATURES_13_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_13_conv_2_out_act_bits;
// --- End of Residual Block 13 ---
add_tensors(unpacked_fm, res_fm_ptr, 6144);

// features.14.conv.0.0 : 8x8x96 -> 8x8x576 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 96, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 96, 8 * 8);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_14_conv_0_0_weights,
    features_14_conv_0_0_pos, features_14_conv_0_0_mask, features_14_conv_0_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 96,
    8, 8, 576,
    1, 1, 0, 1,
    features_14_conv_0_0_scale, features_14_conv_0_0_bias,
    FEATURES_14_CONV_0_0_MAX_D, FEATURES_14_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 36864);
prev_act_bits = features_14_conv_0_0_out_act_bits;

// features.14.conv.1.0 : 8x8x576 -> 4x4x576 (s=2, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 576, 8 * 8);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 8 * 8);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_14_conv_1_0_weights,
    features_14_conv_1_0_pos, features_14_conv_1_0_mask, features_14_conv_1_0_slots,
    prev_act_bits, current_offsets,
    8, 8, 576,
    4, 4,
    3, 3, 1, 2,
    features_14_conv_1_0_scale, features_14_conv_1_0_bias,
    FEATURES_14_CONV_1_0_MAX_D, FEATURES_14_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 9216);
prev_act_bits = features_14_conv_1_0_out_act_bits;

// features.14.conv.2 : 4x4x576 -> 4x4x160 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 576, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 576, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_14_conv_2_weights,
    features_14_conv_2_pos, features_14_conv_2_mask, features_14_conv_2_slots,
    prev_act_bits, current_offsets,
    4, 4, 576,
    4, 4, 160,
    1, 1, 0, 1,
    features_14_conv_2_scale, features_14_conv_2_bias,
    FEATURES_14_CONV_2_MAX_D, FEATURES_14_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_14_conv_2_out_act_bits;

// --- Start of Residual Block 15 ---
for (int i=0; i<2560; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.15.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 160, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 160, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_15_conv_0_0_weights,
    features_15_conv_0_0_pos, features_15_conv_0_0_mask, features_15_conv_0_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 160,
    4, 4, 960,
    1, 1, 0, 1,
    features_15_conv_0_0_scale, features_15_conv_0_0_bias,
    FEATURES_15_CONV_0_0_MAX_D, FEATURES_15_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_15_conv_0_0_out_act_bits;

// features.15.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_15_conv_1_0_weights,
    features_15_conv_1_0_pos, features_15_conv_1_0_mask, features_15_conv_1_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4,
    3, 3, 1, 1,
    features_15_conv_1_0_scale, features_15_conv_1_0_bias,
    FEATURES_15_CONV_1_0_MAX_D, FEATURES_15_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_15_conv_1_0_out_act_bits;

// features.15.conv.2 : 4x4x960 -> 4x4x160 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_15_conv_2_weights,
    features_15_conv_2_pos, features_15_conv_2_mask, features_15_conv_2_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4, 160,
    1, 1, 0, 1,
    features_15_conv_2_scale, features_15_conv_2_bias,
    FEATURES_15_CONV_2_MAX_D, FEATURES_15_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_15_conv_2_out_act_bits;
// --- End of Residual Block 15 ---
add_tensors(unpacked_fm, res_fm_ptr, 2560);

// --- Start of Residual Block 16 ---
for (int i=0; i<2560; i++) res_fm_ptr[i] = unpacked_fm[i];
// features.16.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 160, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 160, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_16_conv_0_0_weights,
    features_16_conv_0_0_pos, features_16_conv_0_0_mask, features_16_conv_0_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 160,
    4, 4, 960,
    1, 1, 0, 1,
    features_16_conv_0_0_scale, features_16_conv_0_0_bias,
    FEATURES_16_CONV_0_0_MAX_D, FEATURES_16_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_16_conv_0_0_out_act_bits;

// features.16.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_16_conv_1_0_weights,
    features_16_conv_1_0_pos, features_16_conv_1_0_mask, features_16_conv_1_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4,
    3, 3, 1, 1,
    features_16_conv_1_0_scale, features_16_conv_1_0_bias,
    FEATURES_16_CONV_1_0_MAX_D, FEATURES_16_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_16_conv_1_0_out_act_bits;

// features.16.conv.2 : 4x4x960 -> 4x4x160 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_16_conv_2_weights,
    features_16_conv_2_pos, features_16_conv_2_mask, features_16_conv_2_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4, 160,
    1, 1, 0, 1,
    features_16_conv_2_scale, features_16_conv_2_bias,
    FEATURES_16_CONV_2_MAX_D, FEATURES_16_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_16_conv_2_out_act_bits;
// --- End of Residual Block 16 ---
add_tensors(unpacked_fm, res_fm_ptr, 2560);

// features.17.conv.0.0 : 4x4x160 -> 4x4x960 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 160, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 160, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_17_conv_0_0_weights,
    features_17_conv_0_0_pos, features_17_conv_0_0_mask, features_17_conv_0_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 160,
    4, 4, 960,
    1, 1, 0, 1,
    features_17_conv_0_0_scale, features_17_conv_0_0_bias,
    FEATURES_17_CONV_0_0_MAX_D, FEATURES_17_CONV_0_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_17_conv_0_0_out_act_bits;

// features.17.conv.1.0 : 4x4x960 -> 4x4x960 (s=1, p=1)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_depthwise_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_17_conv_1_0_weights,
    features_17_conv_1_0_pos, features_17_conv_1_0_mask, features_17_conv_1_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4,
    3, 3, 1, 1,
    features_17_conv_1_0_scale, features_17_conv_1_0_bias,
    FEATURES_17_CONV_1_0_MAX_D, FEATURES_17_CONV_1_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 15360);
prev_act_bits = features_17_conv_1_0_out_act_bits;

// features.17.conv.2 : 4x4x960 -> 4x4x320 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 960, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 960, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_17_conv_2_weights,
    features_17_conv_2_pos, features_17_conv_2_mask, features_17_conv_2_slots,
    prev_act_bits, current_offsets,
    4, 4, 960,
    4, 4, 320,
    1, 1, 0, 1,
    features_17_conv_2_scale, features_17_conv_2_bias,
    FEATURES_17_CONV_2_MAX_D, FEATURES_17_CONV_2_WORDS_PER_FILTER);
prev_act_bits = features_17_conv_2_out_act_bits;

// features.18.0 : 4x4x320 -> 4x4x1280 (s=1, p=0)
compute_packed_layout(prev_act_bits, current_offsets, 320, 4 * 4);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 320, 4 * 4);
run_conv2d_general((const uint8_t*)packed_fm, unpacked_fm, features_18_0_weights,
    features_18_0_pos, features_18_0_mask, features_18_0_slots,
    prev_act_bits, current_offsets,
    4, 4, 320,
    4, 4, 1280,
    1, 1, 0, 1,
    features_18_0_scale, features_18_0_bias,
    FEATURES_18_0_MAX_D, FEATURES_18_0_WORDS_PER_FILTER);
apply_relu(unpacked_fm, 20480);
prev_act_bits = features_18_0_out_act_bits;

// classifier.1
global_average_pool_2d(unpacked_fm, packed_fm, 4, 4, 1280);
// After pool, the data is in packed_fm but unpacked. So we just swap pointers.
int8_t *tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
compute_packed_layout(prev_act_bits, current_offsets, 1280, 1);
pack_feature_map(unpacked_fm, (uint8_t*)packed_fm, prev_act_bits, current_offsets, 1280, 1);
run_linear_general((const uint8_t*)packed_fm, unpacked_fm, classifier_1_weights,
    classifier_1_pos, classifier_1_mask, classifier_1_slots,
    prev_act_bits, current_offsets,
    1280, 10,
    classifier_1_scale, classifier_1_bias,
    CLASSIFIER_1_MAX_D, CLASSIFIER_1_WORDS_PER_FILTER);



    int best_class = 0;
    int best_val = unpacked_fm[0];
    for (int i = 1; i < 10; i++) {
        if (unpacked_fm[i] > best_val) { best_val = unpacked_fm[i]; best_class = i; }
    }
    return best_class;
}
#endif



int main(void) {
    uart_puts("\n============================================\n");
    uart_puts("   REQAP-DNN CIFAR-10 BATCH EVALUATION   \n");
    uart_puts("============================================\n");

    FILE *f = fopen("generated/cifar10_test_10k.bin", "rb");
    if (!f) {
        printf("ERROR: Could not open generated/cifar10_test_10k.bin\n");
        return 1;
    }

    int correct = 0;
    int total = 10000;
    for (int i = 0; i < total; i++) {
        uint8_t label;
        fread(&label, 1, 1, f);
        fread(current_test_image, 1, 3072, f);

        int pred = 0;
#ifdef BASELINE_MODE
        pred = run_full_mobilenet_baseline_forward_pass();
#else
        pred = run_full_mobilenet_packed_forward_pass();
#endif
        if (pred == label) {
            correct++;
        }
        
        if ((i + 1) % 100 == 0) {
            printf("Processed %d / 10000 images. Current Accuracy: %.2f%%\r", (i+1), (float)correct / (i+1) * 100.0f);
            fflush(stdout);
        }
    }
    
    printf("\n\n--------------------------------------------\n");
    printf("FINAL HARDWARE TOP-1 ACCURACY: %.2f%%\n", (float)correct / total * 100.0f);
    printf("--------------------------------------------\n");
    fclose(f);
    return 0;
}
