/*
 * main_vc707_vgg11.c
 * ------------------
 * Bare-metal firmware entry point for VGG-11 on VC707 FPGA.
 * Supports both Baseline 8-bit and Packed MQF forward passes via compile-time flag.
 *
 * Requirements:
 * - riscv64-unknown-elf-gcc (rv64imafd)
 * - Custom boot1.S and baremetal_libc_stubs.c
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "swar_mac.h"

// Compile-time UART macro redirects.
#ifdef __riscv
#define RV_LOG(...) do {} while(0)
#define printf(...) do {} while(0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif
#include "hw_metrics.h"
extern uint64_t total_memory_loads;

// -----------------------------------------------------------------------------
// DIRECT HARDWARE UART LOGGING
// -----------------------------------------------------------------------------
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
    while (*s) putchar(*s++);
}
#endif

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

/* ── Include the VGG-11 Model Headers ───────────────────────────────────────*/
#ifdef BASELINE_MODE
  #include "vgg11_baseline_weights.h"
  #define MODEL_NAME "VGG-11 (Baseline 8-bit)"
#else
  #include "vgg11_packed_weights.h"
  #define MODEL_NAME "VGG-11 (Packed MQF)"
#endif

/* 
 * Global Activation Buffer 
 * Dimensioned for largest intermediate feature map (ResNet-18 L4 Conv2 K=4608)
 */
#define MAX_K_SLOTS 4194304
static int8_t act_buf[MAX_K_SLOTS];

static void fill_activations(int8_t *buf, int len, int8_t val)
{
    int i;
    for (i = 0; i < len; i++)
        buf[i] = val;
}

static void print_result(const char *layer, const char *type,
                          int n_words, int d, int32_t result)
{
    printf("  [%-25s] type=%-11s n_words=%4d  d=%d  result=%d\n",
           layer, type, n_words, d, result);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * VGG-11 tests
 * ═══════════════════════════════════════════════════════════════════════════*/
#ifdef TEST_VGG11
#include <stdlib.h>
#include "inference_ops.h"
#include "act_packing.h"
#include "generated/cifar_test_image.h"

// Ping-Pong Buffers for intermediate feature maps (Max size: 64 channels * 32 * 32 = 65536 bytes)
static int8_t fm_A[65536];
static int8_t fm_B[65536];

static void print_layer_range(const char *name, const int8_t *fm, int size) {
    int8_t min_v = 127;
    int8_t max_v = -128;
    int64_t sum = 0;
    for (int i = 0; i < size; i++) {
        int8_t v = fm[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += abs(v);
    }
    RV_LOG("  [Range check] %s: min=%d, max=%d, mean=%f\n", name, min_v, max_v, (double)sum / size);
}

#ifdef __riscv
extern void uart_print_dec(int64_t v);
#else
#include <stdio.h>
#define uart_print_dec(v) printf("%lld", (long long)(v))
#endif

static void print_vgg11_profiler_summary(const char *mode_name,
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

#ifdef BASELINE_MODE
static void run_full_vgg11_baseline_forward_pass(void)
#else
static void run_full_vgg11_packed_forward_pass(void)
#endif
{
#ifdef BASELINE_MODE
    const char *mode_name = "VGG-11 Baseline 8-bit";
#else
    const char *mode_name = "VGG-11 Packed MQF";
#endif

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
    
    // Define dummy 8-bit offset table for layers that take baseline unpacked inputs
    uint8_t dummy_8_act_bits[512];
    uint32_t dummy_8_offsets[512];
    for (int i = 0; i < 512; i++) dummy_8_act_bits[i] = 8;
    
    // Offset tables for packed layers
#ifndef BASELINE_MODE
    uint32_t features_18_offsets[512];
    compute_packed_layout(features_18_out_act_bits, features_18_offsets, 512, 4*4);
    
    uint32_t features_22_offsets[512];
    compute_packed_layout(features_22_out_act_bits, features_22_offsets, 512, 2*2);
    
    uint32_t features_25_offsets[512];
    compute_packed_layout(features_25_out_act_bits, features_25_offsets, 512, 1*1);
    
    uint32_t classifier_0_offsets[4096];
    uint8_t dummy_4096_act_bits[4096];
    for (int i = 0; i < 4096; i++) dummy_4096_act_bits[i] = 8;
    compute_packed_layout(dummy_4096_act_bits, classifier_0_offsets, 4096, 1);
    
    uint32_t classifier_3_offsets[4096];
    compute_packed_layout(classifier_3_out_act_bits, classifier_3_offsets, 4096, 1);
#endif

    /* --- Layer 1: features.0 (Conv 3->64, 32x32) --- */
    RV_LOG("Executing Layer 1 (Conv2D -> ReLU -> MaxPool)\n");
    run_conv2d_8bit(cifar_image_0, fm_A, features_0_weights,
                    32, 32, 3,
                    32, 32, 64,
                    3, 3, 1, 1,
                    features_0_scale, features_0_bias);
    print_layer_range("features.0 (Conv2d out)", fm_A, 64 * 32 * 32);
    apply_relu(fm_A, 64 * 32 * 32);
    print_layer_range("features.0 (ReLU out)", fm_A, 64 * 32 * 32);
    max_pool_2d(fm_A, fm_B, 32, 32, 64);
    print_layer_range("features.0 (MaxPool out)", fm_B, 64 * 16 * 16);

    /* --- Layer 2: features.4 (Conv 64->128, 16x16) --- */
    RV_LOG("Executing Layer 2 (Conv2D)\n");
    run_conv2d_8bit(fm_B, fm_A, features_4_weights,
                    16, 16, 64,
                    16, 16, 128,
                    3, 3, 1, 1,
                    features_4_scale, features_4_bias);
    print_layer_range("features.4 (Conv2d out)", fm_A, 128 * 16 * 16);
    apply_relu(fm_A, 128 * 16 * 16);
    print_layer_range("features.4 (ReLU out)", fm_A, 128 * 16 * 16);
    max_pool_2d(fm_A, fm_B, 16, 16, 128);
    print_layer_range("features.4 (MaxPool out)", fm_B, 128 * 8 * 8);

    /* --- Layer 3: features.8 (Conv 128->256, 8x8) --- */
    RV_LOG("Executing Layer 3 (Conv2D)\n");
    run_conv2d_8bit(fm_B, fm_A, features_8_weights,
                    8, 8, 128,
                    8, 8, 256,
                    3, 3, 1, 1,
                    features_8_scale, features_8_bias);
    print_layer_range("features.8 (Conv2d out)", fm_A, 256 * 8 * 8);
    apply_relu(fm_A, 256 * 8 * 8);
    print_layer_range("features.8 (ReLU out)", fm_A, 256 * 8 * 8);

    /* --- Layer 4: features.11 (Conv 256->256, 8x8) --- */
    RV_LOG("Executing Layer 4 (Conv2D)\n");
    run_conv2d_8bit(fm_A, fm_B, features_11_weights,
                    8, 8, 256,
                    8, 8, 256,
                    3, 3, 1, 1,
                    features_11_scale, features_11_bias);
    print_layer_range("features.11 (Conv2d out)", fm_B, 256 * 8 * 8);
    apply_relu(fm_B, 256 * 8 * 8);
    print_layer_range("features.11 (ReLU out)", fm_B, 256 * 8 * 8);
    max_pool_2d(fm_B, fm_A, 8, 8, 256);
    print_layer_range("features.11 (MaxPool out)", fm_A, 256 * 4 * 4);

    /* --- Layer 5: features.15 (Conv 256->512, 4x4) --- */
    RV_LOG("Executing Layer 5 (Conv2D)\n");
    run_conv2d_8bit(fm_A, fm_B, features_15_weights,
                    4, 4, 256,
                    4, 4, 512,
                    3, 3, 1, 1,
                    features_15_scale, features_15_bias);
    print_layer_range("features.15 (Conv2d out)", fm_B, 512 * 4 * 4);
    apply_relu(fm_B, 512 * 4 * 4);
    print_layer_range("features.15 (ReLU out)", fm_B, 512 * 4 * 4);

    /* --- Layer 6: features.18 (Conv 512->512, 4x4) --- */
    RV_LOG("Executing Layer 6 (Conv2D)\n");
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, features_18_weights,
                    4, 4, 512,
                    4, 4, 512,
                    3, 3, 1, 1,
                    features_18_scale, features_18_bias);
#else
    compute_packed_layout(dummy_8_act_bits, dummy_8_offsets, 512, 4*4);
    run_conv2d_general((const uint8_t*)fm_B, fm_A, features_18_weights,
                       features_18_pos, features_18_mask, features_18_slots,
                       dummy_8_act_bits, dummy_8_offsets,
                       4, 4, 512,
                       4, 4, 512,
                       3, 3, 1, 1,
                       features_18_scale, features_18_bias,
                       FEATURES_18_MAX_D, FEATURES_18_WORDS_PER_FILTER);
#endif
    print_layer_range("features.18 (Conv2d out)", fm_A, 512 * 4 * 4);
    apply_relu(fm_A, 512 * 4 * 4);
    print_layer_range("features.18 (ReLU out)", fm_A, 512 * 4 * 4);
    max_pool_2d(fm_A, fm_B, 4, 4, 512);
    print_layer_range("features.18 (MaxPool out)", fm_B, 512 * 2 * 2);
#ifndef BASELINE_MODE
    // Pack fm_B into fm_A for the next packed layer
    pack_feature_map(fm_B, (uint8_t*)fm_A, features_18_out_act_bits, features_18_offsets, 512, 2*2);
#endif

    /* --- Layer 7: features.22 (Conv 512->512, 2x2) --- */
    RV_LOG("Executing Layer 7 (Conv2D)\n");
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, features_22_weights,
                    2, 2, 512,
                    2, 2, 512,
                    3, 3, 1, 1,
                    features_22_scale, features_22_bias);
#else
    run_conv2d_general((const uint8_t*)fm_A, fm_B, features_22_weights,
                       features_22_pos, features_22_mask, features_22_slots,
                       features_18_out_act_bits, features_18_offsets,
                       2, 2, 512,
                       2, 2, 512,
                       3, 3, 1, 1,
                       features_22_scale, features_22_bias,
                       FEATURES_22_MAX_D, FEATURES_22_WORDS_PER_FILTER);
#endif
    print_layer_range("features.22 (Conv2d out)", fm_B, 512 * 2 * 2);
    apply_relu(fm_B, 512 * 2 * 2);
    print_layer_range("features.22 (ReLU out)", fm_B, 512 * 2 * 2);
#ifndef BASELINE_MODE
    pack_feature_map(fm_B, (uint8_t*)fm_A, features_22_out_act_bits, features_22_offsets, 512, 2*2);
#endif

    /* --- Layer 8: features.25 (Conv 512->512, 2x2) --- */
    RV_LOG("Executing Layer 8 (Conv2D)\n");
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, features_25_weights,
                    2, 2, 512,
                    2, 2, 512,
                    3, 3, 1, 1,
                    features_25_scale, features_25_bias);
#else
    run_conv2d_general((const uint8_t*)fm_A, fm_B, features_25_weights,
                       features_25_pos, features_25_mask, features_25_slots,
                       features_22_out_act_bits, features_22_offsets,
                       2, 2, 512,
                       2, 2, 512,
                       3, 3, 1, 1,
                       features_25_scale, features_25_bias,
                       FEATURES_25_MAX_D, FEATURES_25_WORDS_PER_FILTER);
#endif
    print_layer_range("features.25 (Conv2d out)", fm_B, 512 * 2 * 2);
    apply_relu(fm_B, 512 * 2 * 2);
    print_layer_range("features.25 (ReLU out)", fm_B, 512 * 2 * 2);
    max_pool_2d(fm_B, fm_A, 2, 2, 512);
    print_layer_range("features.25 (MaxPool out)", fm_A, 512 * 1 * 1);
    // Since classifier.0 is baseline, we leave the output of features.25 unpacked in fm_A!

    /* --- Layer 9: classifier.0 (Linear 512 -> 4096) --- */
    RV_LOG("Executing Layer 9 (Linear)\n");
    // Since classifier.0 is completely 8-bit in JSON, Python scripts fall back to baseline format.
    // However, the input is fm_B which is packed. We MUST unpack it before feeding it to baseline 8-bit!
    // But wait, the previous layer (features.25) packed it! If the next layer is baseline, we shouldn't pack it.
    // Let's just use the baseline call, but wait: features.25 already packed it.
    // Oh, I will just disable packing at the end of Layer 8 since Layer 9 is baseline!
    // I'll leave the baseline call and fix Layer 8.
    run_linear_8bit(fm_A, fm_B, classifier_0_weights,
                    512, 4096,
                    classifier_0_scale, classifier_0_bias);
    print_layer_range("classifier.0 (Linear out)", fm_B, 4096);
    apply_relu(fm_B, 4096);
    print_layer_range("classifier.0 (ReLU out)", fm_B, 4096);

    /* --- Layer 10: classifier.3 (Linear 4096 -> 4096) --- */
    RV_LOG("Executing Layer 10 (Linear)\n");
#ifdef BASELINE_MODE
    run_linear_8bit(fm_B, fm_A, classifier_3_weights,
                    4096, 4096,
                    classifier_3_scale, classifier_3_bias);
    print_layer_range("classifier.3 (Linear out)", fm_A, 4096);
    apply_relu(fm_A, 4096);
    print_layer_range("classifier.3 (ReLU out)", fm_A, 4096);
#else
    // Input is fm_B, but it's UNPACKED 8-bit from classifier.0
    // So we use dummy_4096_act_bits which is all 8s!
    run_linear_general((const uint8_t*)fm_B, fm_A, classifier_3_weights,
                       classifier_3_pos, classifier_3_mask, classifier_3_slots,
                       dummy_4096_act_bits, classifier_0_offsets,
                       4096, 4096,
                       classifier_3_scale, classifier_3_bias,
                       CLASSIFIER_3_MAX_D, CLASSIFIER_3_WORDS_PER_FILTER);
    print_layer_range("classifier.3 (Linear out)", fm_A, 4096);
    apply_relu(fm_A, 4096);
    print_layer_range("classifier.3 (ReLU out)", fm_A, 4096);
    // Do NOT pack since classifier.6 is baseline and needs unpacked 8-bit!
#endif

    /* --- Layer 11: classifier.6 (Linear 4096 -> 10) --- */
    RV_LOG("Executing Layer 11 (Linear - Final)\n");
    run_linear_8bit(fm_A, fm_B, classifier_6_weights,
                    4096, 10,
                    classifier_6_scale, classifier_6_bias);

    printf("\n=== Classification Results ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\n", i, fm_B[i]);
    }
    printf("=== End Forward Pass ===\n\n");

#ifdef __riscv
    uint64_t end_cycles = read_mcycle();
    uint64_t end_instrs = read_minstret();
    uint64_t end_time   = mtime_now();
#endif

    /* compute deltas (works on RISC-V and PC) */
#ifdef __riscv
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;
    uint64_t time_ms      = mtime_elapsed_ms(start_time);
#else
    uint64_t cycles_delta = 0ULL;
    uint64_t instrs_delta = 0ULL;
    uint64_t time_ms      = 0ULL;
#endif

    print_vgg11_profiler_summary(mode_name,
                                 total_memory_loads,
                                 cycles_delta,
                                 instrs_delta,
                                 time_ms
                                 );

#ifdef __riscv
    /* Metric logging offloaded to UART; semihosting disabled for bare-metal target */
#endif
}

#endif /* TEST_VGG11 */


#ifdef TEST_VGG11
static void run_vgg11_unit_tests(void)
{
    int32_t r;
    r = swar_mac_8bit(features_0_weights, act_buf, FEATURES_0_N_WORDS);
    print_result("features.0", "8bit-d1", FEATURES_0_N_WORDS, 1, r);

    r = swar_mac_8bit(features_4_weights, act_buf, FEATURES_4_N_WORDS);
    print_result("features.4", "8bit-d1", FEATURES_4_N_WORDS, 1, r);

    r = swar_mac_8bit(features_8_weights, act_buf, FEATURES_8_N_WORDS);
    print_result("features.8", "8bit-d1", FEATURES_8_N_WORDS, 1, r);

    r = swar_mac_8bit(classifier_0_weights, act_buf, CLASSIFIER_0_N_WORDS);
    print_result("classifier.0", "8bit-d1", CLASSIFIER_0_N_WORDS, 1, r);

#ifndef BASELINE_MODE
    r = swar_mac_general(features_25_weights, act_buf, features_25_pos, features_25_mask, features_25_slots,
                         FEATURES_25_MAX_D, FEATURES_25_WORDS_PER_FILTER);
    print_result("features.25", "general-d8", FEATURES_25_WORDS_PER_FILTER, FEATURES_25_MAX_D, r);
#endif
}
#endif /* TEST_VGG11 */


#ifdef __riscv
extern char __bss_start;
extern char __BSS_END__;
int main(void);

void entry_point(void)
{
    // Clear BSS section
    char *dst = &__bss_start;
    char *end = &__BSS_END__;
    while (dst < end) {
        *dst++ = 0;
    }

    // Call main
    (void)main();
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════*/
static void uart_put_kv64(const char* key, uint64_t val) {
    uart_puts(key);
    uart_puts("=");
    char buf[32];
    int i = 31;
    buf[i--] = '\0';
    if (val == 0) {
        buf[i--] = '0';
    } else {
        while (val > 0) {
            buf[i--] = (val % 10) + '0';
            val /= 10;
        }
    }
    uart_puts(&buf[i + 1]);
    uart_puts("\n");
}

static void uart_put_kv32(const char* key, int32_t val) {
    uart_puts(key);
    uart_puts("=");
    if (val < 0) {
        uart_puts("-");
        val = -val;
    }
    char buf[32];
    int i = 31;
    buf[i--] = '\0';
    if (val == 0) {
        buf[i--] = '0';
    } else {
        while (val > 0) {
            buf[i--] = (val % 10) + '0';
            val /= 10;
        }
    }
    uart_puts(&buf[i + 1]);
    uart_puts("\n");
}

int main(void)
{
#ifndef __riscv
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    // DIRECT HARDWARE UART LOGGING
    uart_puts("\n\n--------------------------------------------\n");
    uart_puts("   STARTING VGG-11 MQF PACKED - VC707 \n");
    uart_puts("--------------------------------------------\n");

    fill_activations(act_buf, MAX_K_SLOTS, 1);

    uart_puts("Activations initialized. Running forward pass...\n");

#ifdef BASELINE_MODE
    run_full_vgg11_baseline_forward_pass();
#else
    run_full_vgg11_packed_forward_pass();
#endif

    uart_puts("\n--------------------------------------------\n");
    uart_puts("   VGG-11 EXECUTION FINISHED SUCCESSFULLY!  \n");
    uart_puts("--------------------------------------------\n");

    // Board is done, hang forever
#ifdef BASELINE_MODE
    uart_puts("===BEGIN_METRICS===\nMODEL=VGG-11 Baseline 8-bit\n");
#else
    uart_puts("===BEGIN_METRICS===\nMODEL=VGG-11 Packed MQF\n");
#endif

#ifdef __riscv
    uart_put_kv64("METRIC_CYCLES",       0);
    uart_put_kv64("METRIC_INSTRUCTIONS", 0);
#else
    uart_put_kv64("METRIC_CYCLES",       0);
    uart_put_kv64("METRIC_INSTRUCTIONS", 0);
#endif
    uart_put_kv64("METRIC_MEMORY_LOADS", total_memory_loads);

    int top1 = 0;
    int32_t top1_score = fm_B[0];
    for (int i = 0; i < 10; i++) {
        if (fm_B[i] > top1_score) {
            top1_score = fm_B[i];
            top1 = i;
        }
        char key[16] = "CLASS_SCORE_X";
        key[12] = '0' + i;
        uart_put_kv32(key, (int32_t)fm_B[i]);
    }
    uart_put_kv32("PRED_TOP1_CLASS",     (int32_t)top1);
    uart_put_kv32("PRED_TOP1_SCORE",     (int32_t)top1_score);
    uart_puts("===END_METRICS===\n");

#ifdef __riscv
    while(1) {}
#endif
    
    return 0;
}
