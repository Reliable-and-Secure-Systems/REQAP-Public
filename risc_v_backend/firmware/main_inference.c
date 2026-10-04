/*
 * main_inference.c
 * ================
 * Task 3 — Test harness for SWAR MAC kernel (all models)
 *
 * Tests all three MAC vector types using the generated packed weights
 * for ResNet-18, AlexNet, and VGG-11.
 *
 * Activation = 1 everywhere so result = sum of quantised weights.
 * Compare output against: python risc_v_backend/verify_output.py --model <name>
 *
 * Build (PC — verify correctness):
 *   ResNet-18:
 *     gcc -std=c99 -O2 -Wall -I. -I generated/ \
 *         main_inference.c swar_mac.c generated/resnet18_packed_weights.cc \
 *         -o test_resnet18 && ./test_resnet18
 *
 *   AlexNet:
 *     gcc -std=c99 -O2 -Wall -I. -I generated/ \
 *         main_inference.c swar_mac.c generated/alexnet_packed_weights.cc \
 *         -DTEST_ALEXNET -o test_alexnet && ./test_alexnet
 *
 *   VGG-11:
 *     gcc -std=c99 -O2 -Wall -I. -I generated/ \
 *         main_inference.c swar_mac.c generated/vgg11_packed_weights.cc \
 *         -DTEST_VGG11 -o test_vgg11 && ./test_vgg11
 *
 * Build (RISC-V cross-compile):
 *   riscv-none-embed-gcc -march=rv32i -mabi=ilp32 -O2 -std=c99 \
 *       -I. -I generated/ \
 *       main_inference.c swar_mac.c generated/resnet18_packed_weights.cc \
 *       -DTEST_RESNET18 -o firmware.elf
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "swar_mac.h"

/* Lightweight logging macro: disable verbose prints on RISC-V to
   avoid semihosting stdout overload during simulation. */
#ifdef __riscv
#define RV_LOG(...) ((void)0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif

#ifdef __riscv
static inline uint64_t read_cycles(void) {
    uint32_t cycle_h1, cycle_l, cycle_h2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(cycle_h1));
        __asm__ volatile ("rdcycle %0" : "=r"(cycle_l));
        __asm__ volatile ("rdcycleh %0" : "=r"(cycle_h2));
    } while (cycle_h1 != cycle_h2);
    return ((uint64_t)cycle_h1 << 32) | cycle_l;
}

static inline uint64_t read_instructions(void) {
    uint32_t inst_h1, inst_l, inst_h2;
    do {
        __asm__ volatile ("rdinstreth %0" : "=r"(inst_h1));
        __asm__ volatile ("rdinstret %0" : "=r"(inst_l));
        __asm__ volatile ("rdinstreth %0" : "=r"(inst_h2));
    } while (inst_h1 != inst_h2);
    return ((uint64_t)inst_h1 << 32) | inst_l;
}
#else
static inline uint64_t read_cycles(void) { return 0; }
static inline uint64_t read_instructions(void) { return 0; }
#endif

/* ── Select which model's header to include ──────────────────────────────────
 * Pass -DTEST_ALEXNET, -DTEST_VGG11, or -DTEST_RESNET18 on the compiler
 * command line. Default (no flag) = ResNet-18.
 */
#if defined(TEST_ALEXNET)
  #include "generated/alexnet_packed_weights.h"
  #define MODEL_NAME "AlexNet"
#elif defined(TEST_VGG11)
  #ifdef BASELINE_MODE
    #include "generated/vgg11_baseline_weights.h"
    #define MODEL_NAME "VGG-11 (Baseline 8-bit)"
  #else
    #include "generated/vgg11_packed_weights.h"
    #define MODEL_NAME "VGG-11 (Packed MQF)"
  #endif
#else
  /* Default: ResNet-18 */
  #define TEST_RESNET18
  #include "generated/resnet18_packed_weights.h"
  #define MODEL_NAME "ResNet-18"
#endif

/* ── Activation buffer ───────────────────────────────────────────────────────
 * Largest layer: ResNet-18 layer4 conv2 K=4608.
 * All activations set to 1 → result = sum(quantised weights).
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
 * ResNet-18 tests
 * ═══════════════════════════════════════════════════════════════════════════*/
#ifdef TEST_RESNET18
static void run_resnet18_tests(void)
{
    int32_t r;

    /* Type 1: 8-bit, d=1 */
    r = swar_mac_8bit(conv1_weights, act_buf, CONV1_N_WORDS);
    print_result("conv1", "8bit-d1", CONV1_N_WORDS, 1, r);

    r = swar_mac_8bit(layer1_0_conv1_weights, act_buf, LAYER1_0_CONV1_N_WORDS);
    print_result("layer1.0.conv1", "8bit-d1", LAYER1_0_CONV1_N_WORDS, 1, r);

    r = swar_mac_8bit(layer1_0_conv3_weights, act_buf, LAYER1_0_CONV3_N_WORDS);
    print_result("layer1.0.conv3", "8bit-d1", LAYER1_0_CONV3_N_WORDS, 1, r);

    printf("\n");

    /* Type 2: 4-bit, d=4 */
    r = swar_mac_4bit_d4(layer1_0_conv2_weights, act_buf, LAYER1_0_CONV2_N_WORDS);
    print_result("layer1.0.conv2", "4bit-d4", LAYER1_0_CONV2_N_WORDS, 4, r);

    r = swar_mac_4bit_d4(layer1_1_conv2_weights, act_buf, LAYER1_1_CONV2_N_WORDS);
    print_result("layer1.1.conv2", "4bit-d4", LAYER1_1_CONV2_N_WORDS, 4, r);

    printf("\n");

    /* Type 3: general (same layer as 4bit-d4 row 1 — must match) */
    r = swar_mac_general(layer1_0_conv2_weights, act_buf,
                          layer1_0_conv2_pos, layer1_0_conv2_mask,
                          4, LAYER1_0_CONV2_MAX_D, LAYER1_0_CONV2_N_WORDS);
    print_result("layer1.0.conv2", "general-d4", LAYER1_0_CONV2_N_WORDS, 4, r);
}
#endif /* TEST_RESNET18 */


/* ═══════════════════════════════════════════════════════════════════════════
 * AlexNet tests
 * ═══════════════════════════════════════════════════════════════════════════*/
#ifdef TEST_ALEXNET
static void run_alexnet_tests(void)
{
    int32_t r;

    /* Type 1: 8-bit, d=1
     * AlexNet config assigns 8-bit to conv1.
     * Deeper conv and FC layers use baseline (K>2000) → still 8-bit, d=1.
     */
    r = swar_mac_8bit(conv1_0_weights, act_buf, CONV1_0_N_WORDS);
    print_result("conv1.0", "8bit-d1", CONV1_0_N_WORDS, 1, r);

    r = swar_mac_8bit(conv2_0_weights, act_buf, CONV2_0_N_WORDS);
    print_result("conv2.0", "8bit-d1", CONV2_0_N_WORDS, 1, r);

    r = swar_mac_8bit(conv3_0_weights, act_buf, CONV3_0_N_WORDS);
    print_result("conv3.0", "8bit-d1", CONV3_0_N_WORDS, 1, r);
}
#endif /* TEST_ALEXNET */


/* ═══════════════════════════════════════════════════════════════════════════
 * VGG-11 tests
 * ═══════════════════════════════════════════════════════════════════════════*/
#ifdef TEST_VGG11
#include <stdlib.h>
#include "inference_ops.h"
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

static void print_vgg11_profiler_summary(const char *mode_name,
                                           uint64_t total_loads,
                                           uint64_t cycles,
                                           uint64_t instrs)
{
    printf("------------------------------------------------------------\n");
    printf("  HARDWARE PROFILER RESULTS (%s)\n", mode_name);
    printf("  Total Memory Loads (lw): %llu\n", (unsigned long long)total_loads);
#ifdef __riscv
    printf("  Total Instruction Count: %llu\n", (unsigned long long)instrs);
    printf("  Total CPU Cycle Count  : %llu\n", (unsigned long long)cycles);
#else
    printf("  [Note: Instruction and Cycle counts are only available on RISC-V]\n");
#endif
    printf("------------------------------------------------------------\n");
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
    uint64_t start_cycles = read_cycles();
    uint64_t start_instrs = read_instructions();
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
    run_conv2d_general(fm_B, fm_A, features_18_weights,
                       features_18_pos, features_18_mask, features_18_slots,
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

    /* --- Layer 7: features.22 (Conv 512->512, 2x2) --- */
    RV_LOG("Executing Layer 7 (Conv2D)\n");
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_B, fm_A, features_22_weights,
                    2, 2, 512,
                    2, 2, 512,
                    3, 3, 1, 1,
                    features_22_scale, features_22_bias);
#else
    run_conv2d_general(fm_B, fm_A, features_22_weights,
                       features_22_pos, features_22_mask, features_22_slots,
                       2, 2, 512,
                       2, 2, 512,
                       3, 3, 1, 1,
                       features_22_scale, features_22_bias,
                       FEATURES_22_MAX_D, FEATURES_22_WORDS_PER_FILTER);
#endif
    print_layer_range("features.22 (Conv2d out)", fm_A, 512 * 2 * 2);
    apply_relu(fm_A, 512 * 2 * 2);
    print_layer_range("features.22 (ReLU out)", fm_A, 512 * 2 * 2);

    /* --- Layer 8: features.25 (Conv 512->512, 2x2) --- */
    RV_LOG("Executing Layer 8 (Conv2D)\n");
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, features_25_weights,
                    2, 2, 512,
                    2, 2, 512,
                    3, 3, 1, 1,
                    features_25_scale, features_25_bias);
#else
    run_conv2d_general(fm_A, fm_B, features_25_weights,
                       features_25_pos, features_25_mask, features_25_slots,
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

    /* --- Layer 9: classifier.0 (Linear 512 -> 4096) --- */
    RV_LOG("Executing Layer 9 (Linear)\n");
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
#else
    run_linear_general(fm_B, fm_A, classifier_3_weights,
                       classifier_3_pos, classifier_3_mask, classifier_3_slots,
                       4096, 4096,
                       classifier_3_scale, classifier_3_bias,
                       CLASSIFIER_3_MAX_D, CLASSIFIER_3_WORDS_PER_FILTER);
#endif
    print_layer_range("classifier.3 (Linear out)", fm_A, 4096);
    apply_relu(fm_A, 4096);
    print_layer_range("classifier.3 (ReLU out)", fm_A, 4096);

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
    uint64_t end_cycles = read_cycles();
    uint64_t end_instrs = read_instructions();
#endif

    /* compute deltas (works on RISC-V and PC) */
#ifdef __riscv
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;
#else
    uint64_t cycles_delta = 0ULL;
    uint64_t instrs_delta = 0ULL;
#endif

    print_vgg11_profiler_summary(mode_name,
                                total_memory_loads,
                                cycles_delta,
                                instrs_delta
                                );

#ifdef __riscv
    /* Write minimal CSV metrics to semihosting file to avoid verbose stdout issues */
#ifdef BASELINE_MODE
    const char *metrics_fname = "metrics_vgg11_baseline.csv";
#else
    const char *metrics_fname = "metrics_vgg11_packed.csv";
#endif
    FILE *mf = fopen(metrics_fname, "w");
    if (mf) {
        fprintf(mf, "METRIC_CYCLES=%llu\n", (unsigned long long)cycles_delta);
        fprintf(mf, "METRIC_INSTRUCTIONS=%llu\n", (unsigned long long)instrs_delta);
        fprintf(mf, "METRIC_MEMORY_LOADS=%llu\n", (unsigned long long)total_memory_loads);
        fclose(mf);
    }
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
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    fill_activations(act_buf, MAX_K_SLOTS, 1);

    printf("\n");
    printf("=============================================================\n");
    printf("  SWAR MAC Kernel Test — %s  (R=16, act=1)\n", MODEL_NAME);
    printf("  Compare: python verify_output.py --model "
#if defined(TEST_ALEXNET)
           "alexnet"
#elif defined(TEST_VGG11)
           "vgg11"
#else
           "resnet18"
#endif
           "\n");
    printf("=============================================================\n\n");

#if defined(TEST_ALEXNET)
    run_alexnet_tests();
#elif defined(TEST_VGG11)
    run_vgg11_unit_tests();
    printf("\n");
#ifdef BASELINE_MODE
    run_full_vgg11_baseline_forward_pass();
#else
    run_full_vgg11_packed_forward_pass();
#endif
#else
    run_resnet18_tests();
    printf("\n");
    printf("  NOTE: Test C (general-d4) must equal Test B row 1 (4bit-d4).\n");
    printf("  If they match, swar_mac_general == swar_mac_4bit_d4  OK.\n");
#endif

    printf("\n=============================================================\n");
    return 0;
}
