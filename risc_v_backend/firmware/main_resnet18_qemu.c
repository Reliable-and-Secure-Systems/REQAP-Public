/* Auto-generated ResNet-18 Execution Harness */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

#include "inference_ops.h"
#include "generated/cifar_test_image.h"

#ifdef BASELINE_MODE
  #include "generated/resnet18_baseline_weights.h"
  #define MODE_NAME "ResNet-18 Baseline 8-bit"
#else
  #include "generated/resnet18_packed_weights.h"
  #define MODE_NAME "ResNet-18 Packed MQF"
#endif

static int8_t fm_A[65536];
static int8_t fm_B[65536];
static int8_t fm_Shortcut[65536];

#ifdef __riscv
#define RV_LOG(...) do {} while(0)
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
#define RV_LOG(...) printf(__VA_ARGS__)
static inline uint64_t read_cycles(void) { return 0; }
static inline uint64_t read_instructions(void) { return 0; }
#endif

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

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\n=== Starting Full %s Forward Pass ===\n", MODE_NAME);
    
    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_cycles();
    uint64_t start_instrs = read_instructions();
#endif
    
    RV_LOG("Executing conv1\n");
    // conv1 is always unpacked 8-bit in both baseline and MQF
    run_conv2d_8bit(cifar_image_0, fm_A, conv1_weights, 32, 32, 3, 32, 32, 64, 3, 3, 1, 1, conv1_scale, conv1_bias);
    apply_relu(fm_A, 64 * 32 * 32);
    print_layer_range("conv1", fm_A, 64*32*32);
    
    RV_LOG("Executing maxpool\n");
    max_pool_3x3_s2_p1(fm_A, fm_B, 32, 32, 64, 16, 16);
    print_layer_range("maxpool", fm_B, 64*16*16);


    RV_LOG("Executing layer1_0\n");
    memcpy(fm_Shortcut, fm_B, 16384);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer1_0_conv1_weights, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_0_conv1_scale, layer1_0_conv1_bias);
    apply_relu(fm_A, 16384);
    print_layer_range("layer1_0.conv1", fm_A, 16384);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer1_0_conv2_weights, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_0_conv2_scale, layer1_0_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer1_0_conv2_weights, layer1_0_conv2_pos, layer1_0_conv2_mask, layer1_0_conv2_slots, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_0_conv2_scale, layer1_0_conv2_bias, LAYER1_0_CONV2_MAX_D, LAYER1_0_CONV2_WORDS_PER_FILTER);
#endif
    

#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_Shortcut, 16384, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_Shortcut, 16384);
#endif
        

    apply_relu(fm_B, 16384);
    print_layer_range("layer1_0 (Add+ReLU)", fm_B, 16384);
    

    RV_LOG("Executing layer1_1\n");
    memcpy(fm_Shortcut, fm_B, 16384);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer1_1_conv1_weights, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_1_conv1_scale, layer1_1_conv1_bias);
    apply_relu(fm_A, 16384);
    print_layer_range("layer1_1.conv1", fm_A, 16384);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer1_1_conv2_weights, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_1_conv2_scale, layer1_1_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer1_1_conv2_weights, layer1_1_conv2_pos, layer1_1_conv2_mask, layer1_1_conv2_slots, 16, 16, 64, 16, 16, 64, 3, 3, 1, 1, layer1_1_conv2_scale, layer1_1_conv2_bias, LAYER1_1_CONV2_MAX_D, LAYER1_1_CONV2_WORDS_PER_FILTER);
#endif
    

#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_Shortcut, 16384, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_Shortcut, 16384);
#endif
        

    apply_relu(fm_B, 16384);
    print_layer_range("layer1_1 (Add+ReLU)", fm_B, 16384);
    

    RV_LOG("Executing layer2_0\n");
    memcpy(fm_Shortcut, fm_B, 16384);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer2_0_conv1_weights, 16, 16, 64, 8, 8, 128, 3, 3, 1, 2, layer2_0_conv1_scale, layer2_0_conv1_bias);
    apply_relu(fm_A, 8192);
    print_layer_range("layer2_0.conv1", fm_A, 8192);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer2_0_conv2_weights, 8, 8, 128, 8, 8, 128, 3, 3, 1, 1, layer2_0_conv2_scale, layer2_0_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer2_0_conv2_weights, layer2_0_conv2_pos, layer2_0_conv2_mask, layer2_0_conv2_slots, 8, 8, 128, 8, 8, 128, 3, 3, 1, 1, layer2_0_conv2_scale, layer2_0_conv2_bias, LAYER2_0_CONV2_MAX_D, LAYER2_0_CONV2_WORDS_PER_FILTER);
#endif
    

    // downsample is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_Shortcut, fm_A, layer2_0_downsample_0_weights, 16, 16, 64, 8, 8, 128, 1, 1, 0, 2, layer2_0_downsample_0_scale, layer2_0_downsample_0_bias);
#else
    run_conv2d_general(fm_Shortcut, fm_A, layer2_0_downsample_0_weights, layer2_0_downsample_0_pos, layer2_0_downsample_0_mask, layer2_0_downsample_0_slots, 16, 16, 64, 8, 8, 128, 1, 1, 0, 2, layer2_0_downsample_0_scale, layer2_0_downsample_0_bias, LAYER2_0_DOWNSAMPLE_0_MAX_D, LAYER2_0_DOWNSAMPLE_0_WORDS_PER_FILTER);
#endif
#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_A, 8192, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_A, 8192);
#endif
        

    apply_relu(fm_B, 8192);
    print_layer_range("layer2_0 (Add+ReLU)", fm_B, 8192);
    

    RV_LOG("Executing layer2_1\n");
    memcpy(fm_Shortcut, fm_B, 8192);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer2_1_conv1_weights, 8, 8, 128, 8, 8, 128, 3, 3, 1, 1, layer2_1_conv1_scale, layer2_1_conv1_bias);
    apply_relu(fm_A, 8192);
    print_layer_range("layer2_1.conv1", fm_A, 8192);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer2_1_conv2_weights, 8, 8, 128, 8, 8, 128, 3, 3, 1, 1, layer2_1_conv2_scale, layer2_1_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer2_1_conv2_weights, layer2_1_conv2_pos, layer2_1_conv2_mask, layer2_1_conv2_slots, 8, 8, 128, 8, 8, 128, 3, 3, 1, 1, layer2_1_conv2_scale, layer2_1_conv2_bias, LAYER2_1_CONV2_MAX_D, LAYER2_1_CONV2_WORDS_PER_FILTER);
#endif
    

#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_Shortcut, 8192, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_Shortcut, 8192);
#endif
        

    apply_relu(fm_B, 8192);
    print_layer_range("layer2_1 (Add+ReLU)", fm_B, 8192);
    

    RV_LOG("Executing layer3_0\n");
    memcpy(fm_Shortcut, fm_B, 8192);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer3_0_conv1_weights, 8, 8, 128, 4, 4, 256, 3, 3, 1, 2, layer3_0_conv1_scale, layer3_0_conv1_bias);
    apply_relu(fm_A, 4096);
    print_layer_range("layer3_0.conv1", fm_A, 4096);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer3_0_conv2_weights, 4, 4, 256, 4, 4, 256, 3, 3, 1, 1, layer3_0_conv2_scale, layer3_0_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer3_0_conv2_weights, layer3_0_conv2_pos, layer3_0_conv2_mask, layer3_0_conv2_slots, 4, 4, 256, 4, 4, 256, 3, 3, 1, 1, layer3_0_conv2_scale, layer3_0_conv2_bias, LAYER3_0_CONV2_MAX_D, LAYER3_0_CONV2_WORDS_PER_FILTER);
#endif
    

    // downsample is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_Shortcut, fm_A, layer3_0_downsample_0_weights, 8, 8, 128, 4, 4, 256, 1, 1, 0, 2, layer3_0_downsample_0_scale, layer3_0_downsample_0_bias);
#else
    run_conv2d_general(fm_Shortcut, fm_A, layer3_0_downsample_0_weights, layer3_0_downsample_0_pos, layer3_0_downsample_0_mask, layer3_0_downsample_0_slots, 8, 8, 128, 4, 4, 256, 1, 1, 0, 2, layer3_0_downsample_0_scale, layer3_0_downsample_0_bias, LAYER3_0_DOWNSAMPLE_0_MAX_D, LAYER3_0_DOWNSAMPLE_0_WORDS_PER_FILTER);
#endif
#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_A, 4096, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_A, 4096);
#endif
        

    apply_relu(fm_B, 4096);
    print_layer_range("layer3_0 (Add+ReLU)", fm_B, 4096);
    

    RV_LOG("Executing layer3_1\n");
    memcpy(fm_Shortcut, fm_B, 4096);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer3_1_conv1_weights, 4, 4, 256, 4, 4, 256, 3, 3, 1, 1, layer3_1_conv1_scale, layer3_1_conv1_bias);
    apply_relu(fm_A, 4096);
    print_layer_range("layer3_1.conv1", fm_A, 4096);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer3_1_conv2_weights, 4, 4, 256, 4, 4, 256, 3, 3, 1, 1, layer3_1_conv2_scale, layer3_1_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer3_1_conv2_weights, layer3_1_conv2_pos, layer3_1_conv2_mask, layer3_1_conv2_slots, 4, 4, 256, 4, 4, 256, 3, 3, 1, 1, layer3_1_conv2_scale, layer3_1_conv2_bias, LAYER3_1_CONV2_MAX_D, LAYER3_1_CONV2_WORDS_PER_FILTER);
#endif
    

#if defined(ENABLE_MQF)
        add_8bit_scaled(fm_B, fm_Shortcut, 4096, 1.0f, 1.0f);
#else
        add_tensors(fm_B, fm_Shortcut, 4096);
#endif
        

    apply_relu(fm_B, 4096);
    print_layer_range("layer3_1 (Add+ReLU)", fm_B, 4096);
    

    RV_LOG("Executing layer4_0\n");
    memcpy(fm_Shortcut, fm_B, 4096);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer4_0_conv1_weights, 4, 4, 256, 2, 2, 512, 3, 3, 1, 2, layer4_0_conv1_scale, layer4_0_conv1_bias);
    apply_relu(fm_A, 2048);
    print_layer_range("layer4_0.conv1", fm_A, 2048);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer4_0_conv2_weights, 2, 2, 512, 2, 2, 512, 3, 3, 1, 1, layer4_0_conv2_scale, layer4_0_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer4_0_conv2_weights, layer4_0_conv2_pos, layer4_0_conv2_mask, layer4_0_conv2_slots, 2, 2, 512, 2, 2, 512, 3, 3, 1, 1, layer4_0_conv2_scale, layer4_0_conv2_bias, LAYER4_0_CONV2_MAX_D, LAYER4_0_CONV2_WORDS_PER_FILTER);
#endif
    

    // downsample is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_Shortcut, fm_A, layer4_0_downsample_0_weights, 4, 4, 256, 2, 2, 512, 1, 1, 0, 2, layer4_0_downsample_0_scale, layer4_0_downsample_0_bias);
#else
    run_conv2d_general(fm_Shortcut, fm_A, layer4_0_downsample_0_weights, layer4_0_downsample_0_pos, layer4_0_downsample_0_mask, layer4_0_downsample_0_slots, 4, 4, 256, 2, 2, 512, 1, 1, 0, 2, layer4_0_downsample_0_scale, layer4_0_downsample_0_bias, LAYER4_0_DOWNSAMPLE_0_MAX_D, LAYER4_0_DOWNSAMPLE_0_WORDS_PER_FILTER);
#endif
    add_tensors(fm_B, fm_A, 2048);
        

    apply_relu(fm_B, 2048);
    print_layer_range("layer4_0 (Add+ReLU)", fm_B, 2048);
    

    RV_LOG("Executing layer4_1\n");
    memcpy(fm_Shortcut, fm_B, 2048);
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit(fm_B, fm_A, layer4_1_conv1_weights, 2, 2, 512, 2, 2, 512, 3, 3, 1, 1, layer4_1_conv1_scale, layer4_1_conv1_bias);
    apply_relu(fm_A, 2048);
    print_layer_range("layer4_1.conv1", fm_A, 2048);
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_A, fm_B, layer4_1_conv2_weights, 2, 2, 512, 2, 2, 512, 3, 3, 1, 1, layer4_1_conv2_scale, layer4_1_conv2_bias);
#else
    run_conv2d_general(fm_A, fm_B, layer4_1_conv2_weights, layer4_1_conv2_pos, layer4_1_conv2_mask, layer4_1_conv2_slots, 2, 2, 512, 2, 2, 512, 3, 3, 1, 1, layer4_1_conv2_scale, layer4_1_conv2_bias, LAYER4_1_CONV2_MAX_D, LAYER4_1_CONV2_WORDS_PER_FILTER);
#endif
    

#ifdef BASELINE_MODE
    add_tensors(fm_B, fm_Shortcut, 2048);
#else
    add_8bit_scaled(fm_B, fm_Shortcut, 2048, 1.0f, 1.0f);
#endif
        

    apply_relu(fm_B, 2048);
    print_layer_range("layer4_1 (Add+ReLU)", fm_B, 2048);
    

    RV_LOG("Executing avgpool\n");
    global_average_pool_2d(fm_B, fm_A, 2, 2, 512);
    print_layer_range("avgpool", fm_A, 512);
    
    RV_LOG("Executing fc\n");
#ifdef BASELINE_MODE
    run_linear_8bit(fm_A, fm_B, fc_weights, 512, 43, fc_scale, fc_bias);
#else
    run_linear_general(fm_A, fm_B, fc_weights, fc_pos, fc_mask, fc_slots, 512, 43, fc_scale, fc_bias, FC_MAX_D, FC_WORDS_PER_FILTER);
#endif
    print_layer_range("fc", fm_B, 43);


    printf("\n=== Classification Results ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\n", i, fm_B[i]);
    }
    printf("=== End Forward Pass ===\n\n");
    
#ifdef __riscv
    uint64_t end_cycles = read_cycles();
    uint64_t end_instrs = read_instructions();
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;
    
    printf("------------------------------------------------------------\n");
    printf("  HARDWARE PROFILER RESULTS (%s)\n", MODE_NAME);
    printf("  Total Memory Loads (lw): %llu\n", (unsigned long long)total_memory_loads);
    printf("  Total Instruction Count: %llu\n", (unsigned long long)instrs_delta);
    printf("  Total CPU Cycle Count  : %llu\n", (unsigned long long)cycles_delta);
    printf("------------------------------------------------------------\n");
    
#ifdef BASELINE_MODE
    uart_puts("===BEGIN_METRICS===\nMODEL=ResNet-18 Baseline 8-bit\n");
#else
    uart_puts("===BEGIN_METRICS===\nMODEL=ResNet-18 Packed MQF\n");
#endif

    uart_put_kv64("METRIC_CYCLES",       cycles_delta);
    uart_put_kv64("METRIC_INSTRUCTIONS", instrs_delta);
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

    /* Spin so QEMU doesn't exit before UART flushes */
    volatile uint32_t spin;
    for (spin = 0; spin < 1000000U; spin++);
#endif
    return 0;
}
