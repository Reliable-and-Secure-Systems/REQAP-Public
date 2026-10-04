/*
 * main_uart_metrics.c
 * ====================
 * RISC-V metrics harness for VGG-11 (Packed MQF and Baseline 8-bit).
 *
 * Replaces semihosting-based printf with direct 16550 UART writes so that
 * QEMU can capture stdout via "-serial file:<path>".
 *
 * Compile (Packed MQF):
 *   riscv-none-elf-gcc -O2 -march=rv32im -mabi=ilp32 -std=c99 \
 *     -I. -I generated/ \
 *     entry.s main_uart_metrics.c swar_mac.c inference_ops.c \
 *     generated/vgg11_packed_weights.cc \
 *     -T link.ld "-Wl,-e,_reset_start" \
 *     -DTEST_VGG11 \
 *     --specs=semihost.specs \
 *     -o firmware_vgg11_packed_uart.elf
 *
 * Compile (Baseline 8-bit):
 *   riscv-none-elf-gcc -O2 -march=rv32im -mabi=ilp32 -std=c99 \
 *     -I. -I generated/ \
 *     entry.s main_uart_metrics.c swar_mac.c inference_ops.c \
 *     generated/vgg11_baseline_weights.cc \
 *     -T link.ld "-Wl,-e,_reset_start" \
 *     -DTEST_VGG11 -DBASELINE_MODE \
 *     --specs=semihost.specs \
 *     -o firmware_vgg11_baseline_uart.elf
 *
 * Run (both):
 *   $job = Start-Job { qemu-system-riscv32 -machine virt -bios none \
 *     -kernel firmware_vgg11_packed_uart.elf -nographic \
 *     -serial file:metrics_packed_uart.txt }
 *   Start-Sleep 300; Stop-Job $job; Remove-Job $job
 */

#include <stdint.h>
#include <string.h>

/* ── UART ─────────────────────────────────────────────────────────────────── */
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

/* Print a uint64 value with a label, e.g. "METRIC_CYCLES=1234567\n" */
static void uart_put_kv64(const char *key, uint64_t v) {
    uart_puts(key);
    uart_putc('=');
    /* Build decimal string in reverse */
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

/* ── RISC-V performance counters ──────────────────────────────────────────── */
static inline uint64_t read_cycles(void) {
    uint32_t h1, l, h2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(h1));
        __asm__ volatile ("rdcycle  %0" : "=r"(l));
        __asm__ volatile ("rdcycleh %0" : "=r"(h2));
    } while (h1 != h2);
    return ((uint64_t)h1 << 32) | l;
}

static inline uint64_t read_instret(void) {
    uint32_t h1, l, h2;
    do {
        __asm__ volatile ("rdinstreth %0" : "=r"(h1));
        __asm__ volatile ("rdinstret  %0" : "=r"(l));
        __asm__ volatile ("rdinstreth %0" : "=r"(h2));
    } while (h1 != h2);
    return ((uint64_t)h1 << 32) | l;
}

/* ── Suppress verbose semihosting printf on RISC-V ─────────────────────────
   We override RV_LOG to nothing, but keep the core inference logic untouched.
   The inference_ops.c diagnostics (first output channel only) will still
   call printf() which goes to semihosting — but we do not rely on it.
   All metrics we care about are emitted via UART below. */

/* ── Include model headers and ops ──────────────────────────────────────────*/
#include "swar_mac.h"
#include "inference_ops.h"

#if defined(TEST_VGG11)
  #ifdef BASELINE_MODE
    #include "generated/vgg11_baseline_weights.h"
    #define MODEL_LABEL  "VGG-11 Baseline 8-bit"
    #define METRICS_TAG  "BASELINE"
  #else
    #include "generated/vgg11_packed_weights.h"
    #define MODEL_LABEL  "VGG-11 Packed MQF"
    #define METRICS_TAG  "PACKED"
  #endif
  #include "generated/cifar_test_image.h"
#else
  #error "Compile with -DTEST_VGG11"
#endif

/* ── Activation ping-pong buffers ────────────────────────────────────────── */
#define MAX_K_SLOTS 4194304
static int8_t act_buf[MAX_K_SLOTS];
static int8_t fm_A[65536];
static int8_t fm_B[65536];

static void fill_activations(int8_t *buf, int len, int8_t val) {
    int i;
    for (i = 0; i < len; i++) buf[i] = val;
}

/* ── Suppressed range-check print (keeps inference_ops happy but silent) ── */
/* inference_ops.c calls printf() for diagnostics — those go through
   semihosting but we tolerate failure there. What we measure is pure
   cycle/instruction counts around the full forward pass. */

/* ── Main entry point ────────────────────────────────────────────────────── */
int main(void) {
    fill_activations(act_buf, MAX_K_SLOTS, 1);

    /* --- Snapshot counters BEFORE forward pass --- */
    uint64_t t0_cyc = read_cycles();
    uint64_t t0_ins = read_instret();

    /* ================================================================
     * Full VGG-11 Forward Pass
     * (identical logic to run_full_vgg11_*_forward_pass in main_inference.c)
     * ================================================================ */
    total_memory_loads = 0;

    /* Layer 1: features.0  Conv(3→64, 32x32, k=3, pad=1) → ReLU → MaxPool */
    run_conv2d_8bit(cifar_image_0, fm_A, features_0_weights,
                    32, 32, 3,
                    32, 32, 64,
                    3, 3, 1, 1,
                    features_0_scale, features_0_bias);
    apply_relu(fm_A, 64 * 32 * 32);
    max_pool_2d(fm_A, fm_B, 32, 32, 64);   /* → 64×16×16 */

    /* Layer 2: features.4  Conv(64→128, 16x16) → ReLU → MaxPool */
    run_conv2d_8bit(fm_B, fm_A, features_4_weights,
                    16, 16, 64,
                    16, 16, 128,
                    3, 3, 1, 1,
                    features_4_scale, features_4_bias);
    apply_relu(fm_A, 128 * 16 * 16);
    max_pool_2d(fm_A, fm_B, 16, 16, 128);  /* → 128×8×8 */

    /* Layer 3: features.8  Conv(128→256, 8x8) → ReLU */
    run_conv2d_8bit(fm_B, fm_A, features_8_weights,
                    8, 8, 128,
                    8, 8, 256,
                    3, 3, 1, 1,
                    features_8_scale, features_8_bias);
    apply_relu(fm_A, 256 * 8 * 8);

    /* Layer 4: features.11 Conv(256→256, 8x8) → ReLU → MaxPool */
    run_conv2d_8bit(fm_A, fm_B, features_11_weights,
                    8, 8, 256,
                    8, 8, 256,
                    3, 3, 1, 1,
                    features_11_scale, features_11_bias);
    apply_relu(fm_B, 256 * 8 * 8);
    max_pool_2d(fm_B, fm_A, 8, 8, 256);    /* → 256×4×4 */

    /* Layer 5: features.15 Conv(256→512, 4x4) → ReLU */
    run_conv2d_8bit(fm_A, fm_B, features_15_weights,
                    4, 4, 256,
                    4, 4, 512,
                    3, 3, 1, 1,
                    features_15_scale, features_15_bias);
    apply_relu(fm_B, 512 * 4 * 4);

    /* Layer 6: features.18 Conv(512→512, 4x4) → ReLU → MaxPool */
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
    apply_relu(fm_A, 512 * 4 * 4);
    max_pool_2d(fm_A, fm_B, 4, 4, 512);    /* → 512×2×2 */

    /* Layer 7: features.22 Conv(512→512, 2x2) → ReLU */
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
    apply_relu(fm_A, 512 * 2 * 2);

    /* Layer 8: features.25 Conv(512→512, 2x2) → ReLU → MaxPool */
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
    apply_relu(fm_B, 512 * 2 * 2);
    max_pool_2d(fm_B, fm_A, 2, 2, 512);    /* → 512×1×1 */

    /* Layer 9: classifier.0 Linear(512→4096) → ReLU
       classifier_0 has MAX_D=1 in both packed and baseline configs → always 8-bit */
    run_linear_8bit(fm_A, fm_B, classifier_0_weights,
                    512, 4096,
                    classifier_0_scale, classifier_0_bias);
    apply_relu(fm_B, 4096);

    /* Layer 10: classifier.3 Linear(4096→4096) → ReLU */
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
    apply_relu(fm_A, 4096);

    /* Layer 11: classifier.6 Linear(4096→10) — final logits */
    run_linear_8bit(fm_A, fm_B, classifier_6_weights,
                    4096, 10,
                    classifier_6_scale, classifier_6_bias);

    /* --- Snapshot counters AFTER forward pass --- */
    uint64_t t1_cyc = read_cycles();
    uint64_t t1_ins = read_instret();

    uint64_t cycles_delta = t1_cyc - t0_cyc;
    uint64_t instrs_delta = t1_ins - t0_ins;

    /* ── Find top-1 class ─────────────────────────────────────────────── */
    int top1 = 0;
    int8_t top1_score = fm_B[0];
    int i;
    for (i = 1; i < 10; i++) {
        if (fm_B[i] > top1_score) {
            top1_score = fm_B[i];
            top1 = i;
        }
    }

    /* ── Emit metrics via UART ─────────────────────────────────────────── */
    uart_puts("===BEGIN_METRICS===\n");
    uart_puts("MODEL=");
    uart_puts(MODEL_LABEL);
    uart_putc('\n');
    uart_put_kv64("METRIC_CYCLES",       cycles_delta);
    uart_put_kv64("METRIC_INSTRUCTIONS", instrs_delta);
    uart_put_kv64("METRIC_MEMORY_LOADS", total_memory_loads);
    uart_put_kv32("PRED_TOP1_CLASS",     (int32_t)top1);
    uart_put_kv32("PRED_TOP1_SCORE",     (int32_t)top1_score);
    /* Print all 10 class scores */
    for (i = 0; i < 10; i++) {
        char key[16] = "CLASS_SCORE_";
        key[12] = '0' + i;
        key[13] = '\0';
        uart_put_kv32(key, (int32_t)fm_B[i]);
    }
    uart_puts("===END_METRICS===\n");

    /* Spin so QEMU doesn't exit before UART flushes */
    volatile uint32_t spin;
    for (spin = 0; spin < 1000000U; spin++);

    return 0;
}
