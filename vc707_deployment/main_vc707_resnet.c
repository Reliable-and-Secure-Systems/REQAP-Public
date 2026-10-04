/* Auto-generated ResNet-18 Execution Wrapper */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "swar_mac.h"
#include "hw_metrics.h"
#include "inference_ops.h"
#include "act_packing.h"
#include "generated/cifar_test_image.h"

#ifdef BASELINE_MODE
  #include "generated/resnet18_baseline_weights.h"
  #define MODEL_NAME "ResNet-18 (Baseline 8-bit)"
#else
  #include "generated/resnet18_packed_weights.h"
  #define MODEL_NAME "ResNet-18 (Packed MQF)"
#endif

#ifdef __riscv
#define RV_LOG(...) do {} while(0)
#define printf(...) do {} while(0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif

#define MAX_K_SLOTS 4194304
static int8_t act_buf[MAX_K_SLOTS];

static int8_t fm_A[65536];
static int8_t fm_B[65536];
static int8_t fm_Res[65536]; // Residual connection buffer

#ifdef __riscv
extern void uart_print_dec(int64_t v);
static inline void uart_putc(char c) {
    volatile uint32_t *UART_TXDATA = (volatile uint32_t *)0x64000000;
    if (c == '\n') {
        while (*UART_TXDATA & 0x80000000);  
        *UART_TXDATA = '\r';
    }
    while (*UART_TXDATA & 0x80000000);  
    *UART_TXDATA = c;
}
static inline void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}
#define CLINT_MTIME (*(volatile uint64_t *)(0x02000000UL + 0xBFF8))
static inline uint64_t mtime_now(void) { return CLINT_MTIME; }
static inline uint64_t mtime_elapsed_ms(uint64_t start) { return ((CLINT_MTIME - start) * 1000) / 100000; }
#else
#define uart_print_dec(v) printf("%lld", (long long)(v))
static inline void uart_putc(char c) { putchar(c); }
static inline void uart_puts(const char *s) { while (*s) putchar(*s++); }
static inline uint64_t mtime_now(void) { return 0; }
static inline uint64_t mtime_elapsed_ms(uint64_t start) { return 0; }
#endif

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

void run_resnet(void) {
    printf("\n=== Starting Full %s Forward Pass ===\n", MODEL_NAME);
    total_memory_loads = 0;
    
    for (int i = 0; i < 3072; i++) {
        fm_A[i] = cifar_image_0[i]; // Load Image
    }
    
    uint8_t dummy_act_bits[512];
    uint32_t current_dummy_offsets[512];
    for (int i = 0; i < 512; i++) dummy_act_bits[i] = 8;

    RV_LOG("Executing conv1\n");
    run_conv2d_8bit(fm_A, fm_B, conv1_weights, 32, 32, 3, 32, 32, 64, 3, 3, 1, 1, conv1_scale, conv1_bias);
    apply_relu(fm_B, 64 * 32 * 32);
    print_layer_range("conv1", fm_B, 64 * 32 * 32);
    memcpy(fm_Res, fm_B, 64 * 32 * 32);
    RV_LOG("Executing layer1_0_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer1_0_conv1_offsets[64];
    compute_packed_layout(layer1_0_conv1_out_act_bits, layer1_0_conv1_offsets, 64, 32*32);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 64, 32*32*1*1);
    run_conv2d_general(fm_B, fm_A, layer1_0_conv1_weights, layer1_0_conv1_pos, layer1_0_conv1_mask, layer1_0_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       32, 32, 64, 32, 32, 64, 3, 3, 1, 1,
                       layer1_0_conv1_scale, layer1_0_conv1_bias, LAYER1_0_CONV1_MAX_D, LAYER1_0_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer1_0_conv1_weights, 32, 32, 64, 32, 32, 64, 3, 3, 1, 1, layer1_0_conv1_scale, layer1_0_conv1_bias);
#endif
    apply_relu(fm_A, 64 * 32 * 32);
    print_layer_range("layer1_0_conv1", fm_A, 64 * 32 * 32);
    RV_LOG("Executing layer1_0_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer1_0_conv2_offsets[64];
    compute_packed_layout(layer1_0_conv2_out_act_bits, layer1_0_conv2_offsets, 64, 32*32);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 64, 32*32*1*1);
    run_conv2d_general(fm_A, fm_B, layer1_0_conv2_weights, layer1_0_conv2_pos, layer1_0_conv2_mask, layer1_0_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       32, 32, 64, 32, 32, 64, 3, 3, 1, 1,
                       layer1_0_conv2_scale, layer1_0_conv2_bias, LAYER1_0_CONV2_MAX_D, LAYER1_0_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer1_0_conv2_weights, 32, 32, 64, 32, 32, 64, 3, 3, 1, 1, layer1_0_conv2_scale, layer1_0_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 64 * 32 * 32);
    print_layer_range("layer1_0_conv2", fm_B, 64 * 32 * 32);
    apply_relu(fm_B, 64 * 32 * 32);
    memcpy(fm_Res, fm_B, 64 * 32 * 32);
    RV_LOG("Executing layer1_1_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer1_1_conv1_offsets[64];
    compute_packed_layout(layer1_1_conv1_out_act_bits, layer1_1_conv1_offsets, 64, 32*32);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 64, 32*32*1*1);
    run_conv2d_general(fm_B, fm_A, layer1_1_conv1_weights, layer1_1_conv1_pos, layer1_1_conv1_mask, layer1_1_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       32, 32, 64, 32, 32, 64, 3, 3, 1, 1,
                       layer1_1_conv1_scale, layer1_1_conv1_bias, LAYER1_1_CONV1_MAX_D, LAYER1_1_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer1_1_conv1_weights, 32, 32, 64, 32, 32, 64, 3, 3, 1, 1, layer1_1_conv1_scale, layer1_1_conv1_bias);
#endif
    apply_relu(fm_A, 64 * 32 * 32);
    print_layer_range("layer1_1_conv1", fm_A, 64 * 32 * 32);
    RV_LOG("Executing layer1_1_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer1_1_conv2_offsets[64];
    compute_packed_layout(layer1_1_conv2_out_act_bits, layer1_1_conv2_offsets, 64, 32*32);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 64, 32*32*1*1);
    run_conv2d_general(fm_A, fm_B, layer1_1_conv2_weights, layer1_1_conv2_pos, layer1_1_conv2_mask, layer1_1_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       32, 32, 64, 32, 32, 64, 3, 3, 1, 1,
                       layer1_1_conv2_scale, layer1_1_conv2_bias, LAYER1_1_CONV2_MAX_D, LAYER1_1_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer1_1_conv2_weights, 32, 32, 64, 32, 32, 64, 3, 3, 1, 1, layer1_1_conv2_scale, layer1_1_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 64 * 32 * 32);
    print_layer_range("layer1_1_conv2", fm_B, 64 * 32 * 32);
    apply_relu(fm_B, 64 * 32 * 32);
    // Downsample path
    run_conv2d_8bit(fm_B, fm_Res, layer2_0_downsample_0_weights, 32, 32, 64, 16, 16, 128, 1, 1, 0, 2, layer2_0_downsample_0_scale, layer2_0_downsample_0_bias);
    RV_LOG("Executing layer2_0_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer2_0_conv1_offsets[128];
    compute_packed_layout(layer2_0_conv1_out_act_bits, layer2_0_conv1_offsets, 128, 16*16);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 64, 16*16*2*2);
    run_conv2d_general(fm_B, fm_A, layer2_0_conv1_weights, layer2_0_conv1_pos, layer2_0_conv1_mask, layer2_0_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       32, 32, 64, 16, 16, 128, 3, 3, 1, 2,
                       layer2_0_conv1_scale, layer2_0_conv1_bias, LAYER2_0_CONV1_MAX_D, LAYER2_0_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer2_0_conv1_weights, 32, 32, 64, 16, 16, 128, 3, 3, 1, 2, layer2_0_conv1_scale, layer2_0_conv1_bias);
#endif
    apply_relu(fm_A, 128 * 16 * 16);
    print_layer_range("layer2_0_conv1", fm_A, 128 * 16 * 16);
    RV_LOG("Executing layer2_0_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer2_0_conv2_offsets[128];
    compute_packed_layout(layer2_0_conv2_out_act_bits, layer2_0_conv2_offsets, 128, 16*16);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 128, 16*16*1*1);
    run_conv2d_general(fm_A, fm_B, layer2_0_conv2_weights, layer2_0_conv2_pos, layer2_0_conv2_mask, layer2_0_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       16, 16, 128, 16, 16, 128, 3, 3, 1, 1,
                       layer2_0_conv2_scale, layer2_0_conv2_bias, LAYER2_0_CONV2_MAX_D, LAYER2_0_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer2_0_conv2_weights, 16, 16, 128, 16, 16, 128, 3, 3, 1, 1, layer2_0_conv2_scale, layer2_0_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 128 * 16 * 16);
    print_layer_range("layer2_0_conv2", fm_B, 128 * 16 * 16);
    apply_relu(fm_B, 128 * 16 * 16);
    memcpy(fm_Res, fm_B, 128 * 16 * 16);
    RV_LOG("Executing layer2_1_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer2_1_conv1_offsets[128];
    compute_packed_layout(layer2_1_conv1_out_act_bits, layer2_1_conv1_offsets, 128, 16*16);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 128, 16*16*1*1);
    run_conv2d_general(fm_B, fm_A, layer2_1_conv1_weights, layer2_1_conv1_pos, layer2_1_conv1_mask, layer2_1_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       16, 16, 128, 16, 16, 128, 3, 3, 1, 1,
                       layer2_1_conv1_scale, layer2_1_conv1_bias, LAYER2_1_CONV1_MAX_D, LAYER2_1_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer2_1_conv1_weights, 16, 16, 128, 16, 16, 128, 3, 3, 1, 1, layer2_1_conv1_scale, layer2_1_conv1_bias);
#endif
    apply_relu(fm_A, 128 * 16 * 16);
    print_layer_range("layer2_1_conv1", fm_A, 128 * 16 * 16);
    RV_LOG("Executing layer2_1_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer2_1_conv2_offsets[128];
    compute_packed_layout(layer2_1_conv2_out_act_bits, layer2_1_conv2_offsets, 128, 16*16);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 128, 16*16*1*1);
    run_conv2d_general(fm_A, fm_B, layer2_1_conv2_weights, layer2_1_conv2_pos, layer2_1_conv2_mask, layer2_1_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       16, 16, 128, 16, 16, 128, 3, 3, 1, 1,
                       layer2_1_conv2_scale, layer2_1_conv2_bias, LAYER2_1_CONV2_MAX_D, LAYER2_1_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer2_1_conv2_weights, 16, 16, 128, 16, 16, 128, 3, 3, 1, 1, layer2_1_conv2_scale, layer2_1_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 128 * 16 * 16);
    print_layer_range("layer2_1_conv2", fm_B, 128 * 16 * 16);
    apply_relu(fm_B, 128 * 16 * 16);
    // Downsample path
    run_conv2d_8bit(fm_B, fm_Res, layer3_0_downsample_0_weights, 16, 16, 128, 8, 8, 256, 1, 1, 0, 2, layer3_0_downsample_0_scale, layer3_0_downsample_0_bias);
    RV_LOG("Executing layer3_0_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer3_0_conv1_offsets[256];
    compute_packed_layout(layer3_0_conv1_out_act_bits, layer3_0_conv1_offsets, 256, 8*8);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 128, 8*8*2*2);
    run_conv2d_general(fm_B, fm_A, layer3_0_conv1_weights, layer3_0_conv1_pos, layer3_0_conv1_mask, layer3_0_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       16, 16, 128, 8, 8, 256, 3, 3, 1, 2,
                       layer3_0_conv1_scale, layer3_0_conv1_bias, LAYER3_0_CONV1_MAX_D, LAYER3_0_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer3_0_conv1_weights, 16, 16, 128, 8, 8, 256, 3, 3, 1, 2, layer3_0_conv1_scale, layer3_0_conv1_bias);
#endif
    apply_relu(fm_A, 256 * 8 * 8);
    print_layer_range("layer3_0_conv1", fm_A, 256 * 8 * 8);
    RV_LOG("Executing layer3_0_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer3_0_conv2_offsets[256];
    compute_packed_layout(layer3_0_conv2_out_act_bits, layer3_0_conv2_offsets, 256, 8*8);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 256, 8*8*1*1);
    run_conv2d_general(fm_A, fm_B, layer3_0_conv2_weights, layer3_0_conv2_pos, layer3_0_conv2_mask, layer3_0_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       8, 8, 256, 8, 8, 256, 3, 3, 1, 1,
                       layer3_0_conv2_scale, layer3_0_conv2_bias, LAYER3_0_CONV2_MAX_D, LAYER3_0_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer3_0_conv2_weights, 8, 8, 256, 8, 8, 256, 3, 3, 1, 1, layer3_0_conv2_scale, layer3_0_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 256 * 8 * 8);
    print_layer_range("layer3_0_conv2", fm_B, 256 * 8 * 8);
    apply_relu(fm_B, 256 * 8 * 8);
    memcpy(fm_Res, fm_B, 256 * 8 * 8);
    RV_LOG("Executing layer3_1_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer3_1_conv1_offsets[256];
    compute_packed_layout(layer3_1_conv1_out_act_bits, layer3_1_conv1_offsets, 256, 8*8);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 256, 8*8*1*1);
    run_conv2d_general(fm_B, fm_A, layer3_1_conv1_weights, layer3_1_conv1_pos, layer3_1_conv1_mask, layer3_1_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       8, 8, 256, 8, 8, 256, 3, 3, 1, 1,
                       layer3_1_conv1_scale, layer3_1_conv1_bias, LAYER3_1_CONV1_MAX_D, LAYER3_1_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer3_1_conv1_weights, 8, 8, 256, 8, 8, 256, 3, 3, 1, 1, layer3_1_conv1_scale, layer3_1_conv1_bias);
#endif
    apply_relu(fm_A, 256 * 8 * 8);
    print_layer_range("layer3_1_conv1", fm_A, 256 * 8 * 8);
    RV_LOG("Executing layer3_1_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer3_1_conv2_offsets[256];
    compute_packed_layout(layer3_1_conv2_out_act_bits, layer3_1_conv2_offsets, 256, 8*8);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 256, 8*8*1*1);
    run_conv2d_general(fm_A, fm_B, layer3_1_conv2_weights, layer3_1_conv2_pos, layer3_1_conv2_mask, layer3_1_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       8, 8, 256, 8, 8, 256, 3, 3, 1, 1,
                       layer3_1_conv2_scale, layer3_1_conv2_bias, LAYER3_1_CONV2_MAX_D, LAYER3_1_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer3_1_conv2_weights, 8, 8, 256, 8, 8, 256, 3, 3, 1, 1, layer3_1_conv2_scale, layer3_1_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 256 * 8 * 8);
    print_layer_range("layer3_1_conv2", fm_B, 256 * 8 * 8);
    apply_relu(fm_B, 256 * 8 * 8);
    // Downsample path
    run_conv2d_8bit(fm_B, fm_Res, layer4_0_downsample_0_weights, 8, 8, 256, 4, 4, 512, 1, 1, 0, 2, layer4_0_downsample_0_scale, layer4_0_downsample_0_bias);
    RV_LOG("Executing layer4_0_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer4_0_conv1_offsets[512];
    compute_packed_layout(layer4_0_conv1_out_act_bits, layer4_0_conv1_offsets, 512, 4*4);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 256, 4*4*2*2);
    run_conv2d_general(fm_B, fm_A, layer4_0_conv1_weights, layer4_0_conv1_pos, layer4_0_conv1_mask, layer4_0_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       8, 8, 256, 4, 4, 512, 3, 3, 1, 2,
                       layer4_0_conv1_scale, layer4_0_conv1_bias, LAYER4_0_CONV1_MAX_D, LAYER4_0_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer4_0_conv1_weights, 8, 8, 256, 4, 4, 512, 3, 3, 1, 2, layer4_0_conv1_scale, layer4_0_conv1_bias);
#endif
    apply_relu(fm_A, 512 * 4 * 4);
    print_layer_range("layer4_0_conv1", fm_A, 512 * 4 * 4);
    RV_LOG("Executing layer4_0_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer4_0_conv2_offsets[512];
    compute_packed_layout(layer4_0_conv2_out_act_bits, layer4_0_conv2_offsets, 512, 4*4);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 512, 4*4*1*1);
    run_conv2d_general(fm_A, fm_B, layer4_0_conv2_weights, layer4_0_conv2_pos, layer4_0_conv2_mask, layer4_0_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       4, 4, 512, 4, 4, 512, 3, 3, 1, 1,
                       layer4_0_conv2_scale, layer4_0_conv2_bias, LAYER4_0_CONV2_MAX_D, LAYER4_0_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer4_0_conv2_weights, 4, 4, 512, 4, 4, 512, 3, 3, 1, 1, layer4_0_conv2_scale, layer4_0_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 512 * 4 * 4);
    print_layer_range("layer4_0_conv2", fm_B, 512 * 4 * 4);
    apply_relu(fm_B, 512 * 4 * 4);
    memcpy(fm_Res, fm_B, 512 * 4 * 4);
    RV_LOG("Executing layer4_1_conv1\n");
#ifndef BASELINE_MODE
    uint32_t layer4_1_conv1_offsets[512];
    compute_packed_layout(layer4_1_conv1_out_act_bits, layer4_1_conv1_offsets, 512, 4*4);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 512, 4*4*1*1);
    run_conv2d_general(fm_B, fm_A, layer4_1_conv1_weights, layer4_1_conv1_pos, layer4_1_conv1_mask, layer4_1_conv1_slots, dummy_act_bits, current_dummy_offsets,
                       4, 4, 512, 4, 4, 512, 3, 3, 1, 1,
                       layer4_1_conv1_scale, layer4_1_conv1_bias, LAYER4_1_CONV1_MAX_D, LAYER4_1_CONV1_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_B, fm_A, layer4_1_conv1_weights, 4, 4, 512, 4, 4, 512, 3, 3, 1, 1, layer4_1_conv1_scale, layer4_1_conv1_bias);
#endif
    apply_relu(fm_A, 512 * 4 * 4);
    print_layer_range("layer4_1_conv1", fm_A, 512 * 4 * 4);
    RV_LOG("Executing layer4_1_conv2\n");
#ifndef BASELINE_MODE
    uint32_t layer4_1_conv2_offsets[512];
    compute_packed_layout(layer4_1_conv2_out_act_bits, layer4_1_conv2_offsets, 512, 4*4);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 512, 4*4*1*1);
    run_conv2d_general(fm_A, fm_B, layer4_1_conv2_weights, layer4_1_conv2_pos, layer4_1_conv2_mask, layer4_1_conv2_slots, dummy_act_bits, current_dummy_offsets,
                       4, 4, 512, 4, 4, 512, 3, 3, 1, 1,
                       layer4_1_conv2_scale, layer4_1_conv2_bias, LAYER4_1_CONV2_MAX_D, LAYER4_1_CONV2_WORDS_PER_FILTER);
    // Unpack logic skipped in hardware test, assuming unpacked array holds outputs.
#else
    run_conv2d_8bit(fm_A, fm_B, layer4_1_conv2_weights, 4, 4, 512, 4, 4, 512, 3, 3, 1, 1, layer4_1_conv2_scale, layer4_1_conv2_bias);
#endif
    add_tensors(fm_B, fm_Res, 512 * 4 * 4);
    print_layer_range("layer4_1_conv2", fm_B, 512 * 4 * 4);
    apply_relu(fm_B, 512 * 4 * 4);

    // AvgPool 4x4 -> 1x1
    global_average_pool_2d(fm_B, fm_A, 4, 4, 512);
    
    // FC layer
    RV_LOG("Executing Layer fc (Linear)\n");
#ifndef BASELINE_MODE
    uint32_t fc_offsets[64];
    compute_packed_layout(fc_out_act_bits, fc_offsets, 64, 1);
    compute_packed_layout(dummy_act_bits, current_dummy_offsets, 512, 1);
    run_linear_general((uint8_t*)fm_A, fm_B, fc_weights, fc_pos, fc_mask, fc_slots, dummy_act_bits, current_dummy_offsets,
                       512, 64, fc_scale, fc_bias, FC_MAX_D, FC_WORDS_PER_FILTER);
#else
    run_linear_8bit(fm_A, fm_B, fc_weights, 512, 64, fc_scale, fc_bias);
#endif

    // Print final logits
    printf("\n=== Classification Results ===\n");
    for (int i = 0; i < 43; i++) {
        if (i >= 10) break; // CIFAR-10 has 10 classes
        printf("Class %d: %d\n", i, fm_B[i]);
    }
}

#ifdef TEST_RESNET18
int main(void) {
    run_resnet();
    return 0;
}
#endif
