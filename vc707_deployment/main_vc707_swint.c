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

    uint8_t dummy_in_act_bits[3] = {8, 8, 8};
    uint32_t dummy_offset_table[3] = {0, 1, 2};

    RV_LOG("Executing Patch Embedding\n");
    // patch_embed.proj is a Conv2d: 3 -> 96 channels, kernel=4, stride=4
    // Output: 96 x 56 x 56 = 301056 elements.
    run_conv2d_general((const uint8_t*)fm_A, fm_B, patch_embed_proj_weights, 
                       patch_embed_proj_pos, patch_embed_proj_mask, patch_embed_proj_slots,
                       dummy_in_act_bits, dummy_offset_table, 
                       224, 224, 3, 56, 56, 96, 4, 4, 0, 4, 
                       patch_embed_proj_scale, patch_embed_proj_bias, 
                       PATCH_EMBED_PROJ_MAX_D, PATCH_EMBED_PROJ_WORDS_PER_FILTER);
                       
    // Swin reshapes output to (B, H*W, C) -> (1, 3136, 96)
    print_layer_range("patch_embed.proj", fm_B, 301056);

    uint8_t dummy_in_act_bits_96[96];
    uint32_t dummy_offset_table_96[96];
    for (int i=0; i<96; i++) { dummy_in_act_bits_96[i]=8; dummy_offset_table_96[i]=i; }

    RV_LOG("Executing Block 0 - Attention QKV\n");
    // Layer 0 Block 0 QKV: 96 -> 288 (Linear layer over 3136 sequence tokens)
    run_linear_general((const uint8_t*)fm_B, fm_A, layers_0_blocks_0_attn_qkv_weights,
                       layers_0_blocks_0_attn_qkv_pos, layers_0_blocks_0_attn_qkv_mask, layers_0_blocks_0_attn_qkv_slots,
                       dummy_in_act_bits_96, dummy_offset_table_96,
                       96, 288, layers_0_blocks_0_attn_qkv_scale, layers_0_blocks_0_attn_qkv_bias,
                       LAYERS_0_BLOCKS_0_ATTN_QKV_MAX_D, LAYERS_0_BLOCKS_0_ATTN_QKV_WORDS_PER_FILTER);

    // After QKV, it gets split, attention is computed, and then PROJ is applied
    // (Skipping the raw SWAR attention invocation here to keep it simplified for hardware testing)
    
    RV_LOG("Executing Block 0 - Attention PROJ\n");
    // Attention PROJ: 96 -> 96
    run_linear_general((const uint8_t*)fm_A, fm_B, layers_0_blocks_0_attn_proj_weights,
                       layers_0_blocks_0_attn_proj_pos, layers_0_blocks_0_attn_proj_mask, layers_0_blocks_0_attn_proj_slots,
                       dummy_in_act_bits_96, dummy_offset_table_96,
                       96, 96, layers_0_blocks_0_attn_proj_scale, layers_0_blocks_0_attn_proj_bias,
                       LAYERS_0_BLOCKS_0_ATTN_PROJ_MAX_D, LAYERS_0_BLOCKS_0_ATTN_PROJ_WORDS_PER_FILTER);

    RV_LOG("Executing Block 0 - MLP FC1\n");
    // MLP FC1: 96 -> 384
    run_linear_general((const uint8_t*)fm_B, fm_A, layers_0_blocks_0_mlp_fc1_weights,
                       layers_0_blocks_0_mlp_fc1_pos, layers_0_blocks_0_mlp_fc1_mask, layers_0_blocks_0_mlp_fc1_slots,
                       dummy_in_act_bits_96, dummy_offset_table_96,
                       96, 384, layers_0_blocks_0_mlp_fc1_scale, layers_0_blocks_0_mlp_fc1_bias,
                       LAYERS_0_BLOCKS_0_MLP_FC1_MAX_D, LAYERS_0_BLOCKS_0_MLP_FC1_WORDS_PER_FILTER);

    // apply_gelu(fm_A, 3136 * 384);
    
    uint8_t dummy_in_act_bits_384[384];
    uint32_t dummy_offset_table_384[384];
    for (int i=0; i<384; i++) { dummy_in_act_bits_384[i]=8; dummy_offset_table_384[i]=i; }

    RV_LOG("Executing Block 0 - MLP FC2\n");
    // MLP FC2: 384 -> 96
    run_linear_general((const uint8_t*)fm_A, fm_B, layers_0_blocks_0_mlp_fc2_weights,
                       layers_0_blocks_0_mlp_fc2_pos, layers_0_blocks_0_mlp_fc2_mask, layers_0_blocks_0_mlp_fc2_slots,
                       dummy_in_act_bits_384, dummy_offset_table_384,
                       384, 96, layers_0_blocks_0_mlp_fc2_scale, layers_0_blocks_0_mlp_fc2_bias,
                       LAYERS_0_BLOCKS_0_MLP_FC2_MAX_D, LAYERS_0_BLOCKS_0_MLP_FC2_WORDS_PER_FILTER);

    printf("\n=== Execution Complete (Swin Block 0 Test) ===\n");
    
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
