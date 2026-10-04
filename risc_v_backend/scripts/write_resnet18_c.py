import os

lines = []
lines.append('''/* Auto-generated ResNet-18 Execution Harness */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "inference_ops.h"
#include "act_packing.h"
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

// -----------------------------------------------------------------------------
// DIRECT HARDWARE UART LOGGING
// -----------------------------------------------------------------------------
#ifdef __riscv
#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))

static void uart_putc(char c) {
    if (c == '\\n') {
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\\r';
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
#else
static void uart_putc(char c) { printf("%c", c); }
static void uart_puts(const char *s) { printf("%s", s); }
static inline uint64_t mtime_now(void) { return 0; }
static inline uint64_t mtime_elapsed_ms(uint64_t start) { return 0; }
#endif

extern void uart_print_dec(int64_t v);

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
    uart_putc('\\n');
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
    uart_putc('\\n');
}


#ifdef __riscv
#define RV_LOG(...) do {} while(0)
#else
#define RV_LOG(...) printf(__VA_ARGS__)
#endif
#include "hw_metrics.h"

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
    RV_LOG("  [Range check] %s: min=%d, max=%d, mean=%f\\n", name, min_v, max_v, (double)sum / size);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\\n=== Starting Full %s Forward Pass ===\\n", MODE_NAME);
    
    total_memory_loads = 0;
#ifdef __riscv
    uint64_t start_cycles = read_mcycle();
    uint64_t start_instrs = read_minstret();
    uint64_t start_time   = mtime_now();
#endif

    uint32_t current_offsets[65536];
    const uint8_t *prev_act_bits = NULL;
    
    RV_LOG("Executing conv1\\n");
    // conv1 is always unpacked 8-bit in both baseline and MQF
    run_conv2d_8bit(cifar_image_0, fm_A, conv1_weights, 32, 32, 3, 32, 32, 64, 3, 3, 1, 1, conv1_scale, conv1_bias);
    apply_relu(fm_A, 64 * 32 * 32);
    print_layer_range("conv1", fm_A, 64*32*32);
    
    RV_LOG("Executing maxpool\\n");
    max_pool_3x3_s2_p1(fm_A, fm_B, 32, 32, 64, 16, 16);
    print_layer_range("maxpool", fm_B, 64*16*16);
''')

stages = [
    (1, 0, 64, 64, 16, 16, 1, False),
    (1, 1, 64, 64, 16, 16, 1, False),
    (2, 0, 64, 128, 16, 8, 2, True),
    (2, 1, 128, 128, 8, 8, 1, False),
    (3, 0, 128, 256, 8, 4, 2, True),
    (3, 1, 256, 256, 4, 4, 1, False),
    (4, 0, 256, 512, 4, 2, 2, True),
    (4, 1, 512, 512, 2, 2, 1, False),
]

# We are at fm_B after maxpool
current_buf = "fm_B"
next_buf = "fm_A"
for stage, block, in_ch, out_ch, in_size, out_size, stride, downsample in stages:
    prefix = f"layer{stage}_{block}"
    
    lines.append(f'''
    RV_LOG("Executing {prefix}\\n");
    memcpy(fm_Shortcut, {current_buf}, {in_ch * in_size * in_size});
    
    // conv1 in all basic blocks is 8-bit unpacked
    run_conv2d_8bit({current_buf}, {next_buf}, {prefix}_conv1_weights, {in_size}, {in_size}, {in_ch}, {out_size}, {out_size}, {out_ch}, 3, 3, 1, {stride}, {prefix}_conv1_scale, {prefix}_conv1_bias);
    apply_relu({next_buf}, {out_ch * out_size * out_size});
    print_layer_range("{prefix}.conv1", {next_buf}, {out_ch * out_size * out_size});
    
    // conv2 is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit({next_buf}, {current_buf}, {prefix}_conv2_weights, {out_size}, {out_size}, {out_ch}, {out_size}, {out_size}, {out_ch}, 3, 3, 1, 1, {prefix}_conv2_scale, {prefix}_conv2_bias);
#else
    prev_act_bits = {prefix}_conv1_out_act_bits;
    compute_packed_layout(prev_act_bits, current_offsets, {out_ch}, {out_size * out_size});
    pack_feature_map({next_buf}, (uint8_t*){current_buf}, prev_act_bits, current_offsets, {out_ch}, {out_size * out_size});
    run_conv2d_general((const uint8_t*){current_buf}, {next_buf}, {prefix}_conv2_weights, {prefix}_conv2_pos, {prefix}_conv2_mask, {prefix}_conv2_slots, prev_act_bits, current_offsets, {out_size}, {out_size}, {out_ch}, {out_size}, {out_size}, {out_ch}, 3, 3, 1, 1, {prefix}_conv2_scale, {prefix}_conv2_bias, {prefix.upper()}_CONV2_MAX_D, {prefix.upper()}_CONV2_WORDS_PER_FILTER);
    // Move un-packed result from next_buf back to current_buf for residual addition
    memcpy({current_buf}, {next_buf}, {out_ch * out_size * out_size});
#endif
    ''')
    
    if downsample:
        lines.append(f'''
    // downsample is packed
#ifdef BASELINE_MODE
    run_conv2d_8bit(fm_Shortcut, {next_buf}, {prefix}_downsample_0_weights, {in_size}, {in_size}, {in_ch}, {out_size}, {out_size}, {out_ch}, 1, 1, 0, {stride}, {prefix}_downsample_0_scale, {prefix}_downsample_0_bias);
#else
    // Use the output bit width of the PREVIOUS layer for the downsample
    // The previous layer for the downsample is actually the input to the block, 
    // but downsample uses 8-bit input in the ResNet architecture here because the block input is 8-bit!
    // We can simply pass the same 8-bit dummy array.
    prev_act_bits = {prefix}_conv1_out_act_bits; // It's an array of 8s, perfect for downsample!
    compute_packed_layout(prev_act_bits, current_offsets, {in_ch}, {in_size * in_size});
    pack_feature_map(fm_Shortcut, (uint8_t*){next_buf}, prev_act_bits, current_offsets, {in_ch}, {in_size * in_size});
    run_conv2d_general((const uint8_t*){next_buf}, fm_Shortcut, {prefix}_downsample_0_weights, {prefix}_downsample_0_pos, {prefix}_downsample_0_mask, {prefix}_downsample_0_slots, prev_act_bits, current_offsets, {in_size}, {in_size}, {in_ch}, {out_size}, {out_size}, {out_ch}, 1, 1, 0, {stride}, {prefix}_downsample_0_scale, {prefix}_downsample_0_bias, {prefix.upper()}_DOWNSAMPLE_0_MAX_D, {prefix.upper()}_DOWNSAMPLE_0_WORDS_PER_FILTER);
    // Put unpacked downsampled fm_Shortcut into next_buf so it matches the non-downsample flow
    memcpy({next_buf}, fm_Shortcut, {out_ch * out_size * out_size});
#endif
    add_tensors({current_buf}, {next_buf}, {out_ch * out_size * out_size});
        ''')
    else:
        lines.append(f'''
    add_tensors({current_buf}, fm_Shortcut, {out_ch * out_size * out_size});
        ''')
        
    lines.append(f'''
    apply_relu({current_buf}, {out_ch * out_size * out_size});
    print_layer_range("{prefix} (Add+ReLU)", {current_buf}, {out_ch * out_size * out_size});
    ''')

lines.append(f'''
    RV_LOG("Executing avgpool\\n");
    global_average_pool_2d({current_buf}, {next_buf}, 2, 2, 512);
    print_layer_range("avgpool", {next_buf}, 512);
    
    RV_LOG("Executing fc\\n");
#ifdef BASELINE_MODE
    run_linear_8bit({next_buf}, {current_buf}, fc_weights, 512, 10, fc_scale, fc_bias);
#else
    prev_act_bits = layer4_1_conv2_out_act_bits; // The FC layer takes the output bits of the last conv2
    compute_packed_layout(prev_act_bits, current_offsets, 512, 1);
    pack_feature_map({next_buf}, (uint8_t*){current_buf}, prev_act_bits, current_offsets, 512, 1);
    run_linear_general((const uint8_t*){current_buf}, {next_buf}, fc_weights, fc_pos, fc_mask, fc_slots, prev_act_bits, current_offsets, 512, 10, fc_scale, fc_bias, FC_MAX_D, FC_WORDS_PER_FILTER);
    // Result is in next_buf, move back to current_buf for final output
    memcpy({current_buf}, {next_buf}, 10);
#endif
    print_layer_range("fc", {current_buf}, 10);
''')

lines.append('''
    printf("\\n=== Classification Results ===\\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: %d\\n", i, fm_B[i]);
    }
    printf("=== End Forward Pass ===\\n\\n");
    
#ifdef __riscv
    uint64_t end_cycles = read_mcycle();
    uint64_t end_instrs = read_minstret();
    uint64_t end_time   = mtime_now();
    
    uint64_t cycles_delta = end_cycles - start_cycles;
    uint64_t instrs_delta = end_instrs - start_instrs;
    uint64_t time_ms      = mtime_elapsed_ms(start_time);
    
    uart_puts("------------------------------------------------------------\\n");
    uart_puts("  HARDWARE PROFILER RESULTS (%s)\\n", MODE_NAME);
    uart_puts("  Total Memory Loads (lw): ");
    uart_print_dec(total_memory_loads);
    uart_puts("\\n");
    
    uart_puts("  Total Instruction Count: ");
    uart_print_dec(instrs_delta);
    uart_puts("\\n");
    
    uart_puts("  Total CPU Cycle Count  : ");
    uart_print_dec(cycles_delta);
    uart_puts("\\n");

    uart_puts("  Total Execution Time   : ");
    uart_print_dec(time_ms);
    uart_puts(" ms\\n");
    
    uart_puts("------------------------------------------------------------\\n");
#endif    

#ifdef BASELINE_MODE
    uart_puts("===BEGIN_METRICS===\\nMODEL=ResNet-18 Baseline 8-bit\\n");
#else
    uart_puts("===BEGIN_METRICS===\\nMODEL=ResNet-18 Packed MQF\\n");
#endif

#ifdef __riscv
    uart_put_kv64("METRIC_CYCLES",       cycles_delta);
    uart_put_kv64("METRIC_INSTRUCTIONS", instrs_delta);
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
    uart_puts("===END_METRICS===\\n");

    /* Spin so QEMU doesn't exit before UART flushes */
    volatile uint32_t spin;
    for (spin = 0; spin < 1000000U; spin++);
    return 0;
}
''')

with open('vc707_deployment/main_vc707_resnet18.c', 'w') as f:
    f.write('\n'.join(lines))
