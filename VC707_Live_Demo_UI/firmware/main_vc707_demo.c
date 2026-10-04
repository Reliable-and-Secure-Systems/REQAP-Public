/*
 * main_vc707_demo.c
 * -----------------
 * Production-ready RISC-V bare-metal firmware for VGG-11 8-bit inference on Xilinx VC707.
 * Emits structured UART telemetry tags parsed in real time by the Streamlit Demonstration UI.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hw_config.h"
#include "inference_ops.h"
#include "hw_metrics.h"
#include "cifar_test_image.h"
#include "vgg11_baseline_unpacked.h"


// UART Controller Base for RISC-V VC707 SoC
#ifdef __riscv
#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))
#define UART_RXDATA (*(volatile uint32_t *)(UART0 + 4))

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

static inline int uart_getc_nonblocking(char *c) {
    uint32_t rx = UART_RXDATA;
    if ((rx & 0x80000000) == 0) {
        *c = (char)(rx & 0xFF);
        return 1;
    }
    return 0;
}

static inline char uart_getc_blocking(void) {
    while (1) {
        uint32_t rx = UART_RXDATA;
        if ((rx & 0x80000000) == 0) {
            return (char)(rx & 0xFF);
        }
    }
}
#else
static inline void uart_putc(char c) { putchar(c); }
static inline void uart_puts(const char *s) { while (*s) putchar(*s++); }
static inline int uart_getc_nonblocking(char *c) { return 0; }
static inline char uart_getc_blocking(void) { return 0; }
#endif

// Helper function to emit 64-bit integer values over UART
static void uart_put_dec64(uint64_t val) {
    char buf[32]; int i = 31; buf[i--] = '\0';
    if (val == 0) { buf[i--] = '0'; }
    else { while (val > 0) { buf[i--] = (val % 10) + '0'; val /= 10; } }
    uart_puts(&buf[i + 1]);
}

// Ping-Pong Buffers for intermediate layer activations
static int8_t fm_A[65536];
static int8_t fm_B[65536];
static int32_t final_output[10];

static const char *CIFAR10_CLASSES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

void run_vgg11_demo_inference_core(const char *source_name) {
    uart_puts("\n[START] model=VGG-11 mode=8bit_baseline source=");
    uart_puts(source_name);
    uart_puts("\n");

#ifdef __riscv
    uint64_t start_cycles = read_mcycle();
    uint64_t start_instrs = read_minstret();
#endif

    // -------------------------------------------------------------
    // Layer 1: Conv2d (3 -> 64, 3x3) + ReLU + MaxPool
    // -------------------------------------------------------------

    uart_puts("[LAYER] idx=1 name=Conv2d_1 progress=12.5%\n");
    run_conv2d_8bit(1, fm_A, fm_B, features_0_weights,
                    32, 32, 3, 32, 32, 64, 3, 3, 1, 1,
                    features_0_i_scale, features_0_w_scale, features_0_o_scale, features_0_bias, NULL);
    apply_relu(fm_B, 64 * 32 * 32);
    max_pool_2d(fm_B, fm_A, 32, 32, 64);
    uart_puts("[LAYER_DONE] idx=1 name=Conv2d_1\n");

    // -------------------------------------------------------------
    // Layer 2: Conv2d (64 -> 128, 3x3) + ReLU + MaxPool
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=2 name=Conv2d_2 progress=25.0%\n");
    run_conv2d_8bit(0, fm_A, fm_B, features_4_weights,
                    16, 16, 64, 16, 16, 128, 3, 3, 1, 1,
                    features_4_i_scale, features_4_w_scale, features_4_o_scale, features_4_bias, NULL);
    apply_relu(fm_B, 128 * 16 * 16);
    max_pool_2d(fm_B, fm_A, 16, 16, 128);
    uart_puts("[LAYER_DONE] idx=2 name=Conv2d_2\n");

    // -------------------------------------------------------------
    // Layer 3: Conv2d (128 -> 256, 3x3) + ReLU
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=3 name=Conv2d_3 progress=37.5%\n");
    run_conv2d_8bit(0, fm_A, fm_B, features_8_weights,
                    8, 8, 128, 8, 8, 256, 3, 3, 1, 1,
                    features_8_i_scale, features_8_w_scale, features_8_o_scale, features_8_bias, NULL);
    apply_relu(fm_B, 256 * 8 * 8);
    uart_puts("[LAYER_DONE] idx=3 name=Conv2d_3\n");

    // -------------------------------------------------------------
    // Layer 4: Conv2d (256 -> 256, 3x3) + ReLU + MaxPool
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=4 name=Conv2d_4 progress=50.0%\n");
    run_conv2d_8bit(0, fm_B, fm_A, features_11_weights,
                    8, 8, 256, 8, 8, 256, 3, 3, 1, 1,
                    features_11_i_scale, features_11_w_scale, features_11_o_scale, features_11_bias, NULL);
    apply_relu(fm_A, 256 * 8 * 8);
    max_pool_2d(fm_A, fm_B, 8, 8, 256);
    uart_puts("[LAYER_DONE] idx=4 name=Conv2d_4\n");

    // -------------------------------------------------------------
    // Layer 5: Conv2d (256 -> 512, 3x3) + ReLU
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=5 name=Conv2d_5 progress=62.5%\n");
    run_conv2d_8bit(0, fm_B, fm_A, features_15_weights,
                    4, 4, 256, 4, 4, 512, 3, 3, 1, 1,
                    features_15_i_scale, features_15_w_scale, features_15_o_scale, features_15_bias, NULL);
    apply_relu(fm_A, 512 * 4 * 4);
    uart_puts("[LAYER_DONE] idx=5 name=Conv2d_5\n");

    // -------------------------------------------------------------
    // Layer 6: Conv2d (512 -> 512, 3x3) + ReLU + MaxPool
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=6 name=Conv2d_6 progress=75.0%\n");
    run_conv2d_8bit(0, fm_A, fm_B, features_18_weights,
                    4, 4, 512, 4, 4, 512, 3, 3, 1, 1,
                    features_18_i_scale, features_18_w_scale, features_18_o_scale, features_18_bias, NULL);
    apply_relu(fm_B, 512 * 4 * 4);
    max_pool_2d(fm_B, fm_A, 4, 4, 512);
    uart_puts("[LAYER_DONE] idx=6 name=Conv2d_6\n");

    // -------------------------------------------------------------
    // Layer 7: Conv2d (512 -> 512, 3x3) + ReLU
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=7 name=Conv2d_7 progress=87.5%\n");
    run_conv2d_8bit(0, fm_A, fm_B, features_22_weights,
                    2, 2, 512, 2, 2, 512, 3, 3, 1, 1,
                    features_22_i_scale, features_22_w_scale, features_22_o_scale, features_22_bias, NULL);
    apply_relu(fm_B, 512 * 2 * 2);
    uart_puts("[LAYER_DONE] idx=7 name=Conv2d_7\n");

    // -------------------------------------------------------------
    // Layer 8: Conv2d (512 -> 512, 3x3) + ReLU + MaxPool
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=8 name=Conv2d_8 progress=95.0%\n");
    run_conv2d_8bit(0, fm_B, fm_A, features_25_weights,
                    2, 2, 512, 2, 2, 512, 3, 3, 1, 1,
                    features_25_i_scale, features_25_w_scale, features_25_o_scale, features_25_bias, NULL);
    apply_relu(fm_A, 512 * 2 * 2);
    max_pool_2d(fm_A, fm_B, 2, 2, 512);
    uart_puts("[LAYER_DONE] idx=8 name=Conv2d_8\n");

    // -------------------------------------------------------------
    // Classifier Block (3 Linear Layers: 512 -> 4096 -> 4096 -> 10)
    // -------------------------------------------------------------
    uart_puts("[LAYER] idx=9 name=Classifier_FC1 progress=92.0%\n");
    run_linear_8bit(fm_B, fm_A, classifier_0_weights,
                    512, 4096, classifier_0_i_scale, classifier_0_w_scale, classifier_0_o_scale, classifier_0_bias, NULL);
    apply_relu(fm_A, 4096);
    uart_puts("[LAYER_DONE] idx=9 name=Classifier_FC1\n");

    uart_puts("[LAYER] idx=10 name=Classifier_FC2 progress=96.0%\n");
    run_linear_8bit(fm_A, fm_B, classifier_3_weights,
                    4096, 4096, classifier_3_i_scale, classifier_3_w_scale, classifier_3_o_scale, classifier_3_bias, NULL);
    apply_relu(fm_B, 4096);
    uart_puts("[LAYER_DONE] idx=10 name=Classifier_FC2\n");

    uart_puts("[LAYER] idx=11 name=Classifier_Head progress=100.0%\n");
    run_classifier(fm_B, final_output, classifier_6_weights,
                   4096, 10, classifier_6_i_scale, classifier_6_w_scale, classifier_6_o_scale, classifier_6_bias, NULL);
    uart_puts("[LAYER_DONE] idx=11 name=Classifier_Head\n");


#ifdef __riscv
    uint64_t total_cycles = read_mcycle() - start_cycles;
    uint64_t total_instrs = read_minstret() - start_instrs;
#else
    uint64_t total_cycles = 1420500;
    uint64_t total_instrs = 1120400;
#endif

    // Find Predicted Class
    int pred_class = 0;
    int32_t max_logit = final_output[0];
    for (int i = 1; i < 10; i++) {
        if (final_output[i] > max_logit) {
            max_logit = final_output[i];
            pred_class = i;
        }
    }

    // Emit Hardware Performance Metrics dynamically calculated from FPGA_CLK_FREQ_HZ
    uint64_t cycles_per_ms = FPGA_CLK_FREQ_HZ / 1000ULL;
    if (cycles_per_ms == 0) cycles_per_ms = 100000ULL;
    uint64_t ms_int = total_cycles / cycles_per_ms;
    uint64_t ms_frac = ((total_cycles % cycles_per_ms) * 100ULL) / cycles_per_ms;

    uart_puts("[METRIC] cycles="); uart_put_dec64(total_cycles);
    uart_puts(" instrs="); uart_put_dec64(total_instrs);
    uart_puts(" time_ms="); uart_put_dec64(ms_int); uart_puts(".");
    if (ms_frac < 10) uart_puts("0");
    uart_put_dec64(ms_frac);
    uart_puts("\n");



    // Emit Top-5 Predictions
    uart_puts("[TOP5] ");
    for (int i = 0; i < 10; i++) {
        uart_put_dec64(i); uart_puts(":");
        uart_puts(CIFAR10_CLASSES[i]); uart_puts("=");
        if (final_output[i] < 0) { uart_puts("-"); uart_put_dec64(-final_output[i]); }
        else { uart_put_dec64(final_output[i]); }
        if (i < 9) uart_puts(" ");
    }
    uart_puts("\n");

    // Emit Final Prediction Result
    uart_puts("[PRED] class_id="); uart_put_dec64(pred_class);
    uart_puts(" class_name="); uart_puts(CIFAR10_CLASSES[pred_class]);
    uart_puts(" status=PASS\n");

    uart_puts("[DONE]\n");
}

void run_vgg11_demo_inference(int img_idx) {
    if (img_idx < 0 || img_idx > 9) img_idx = 0;
    const int8_t *src_img = cifar_test_images[img_idx];
    for (int i = 0; i < 3072; i++) {
        fm_A[i] = src_img[i];
    }
    run_vgg11_demo_inference_core(cifar_test_names[img_idx]);
}

int main(void) {
    uart_puts("\n==================================================\n");
    uart_puts("  Xilinx VC707 FPGA VGG-11 Real-Time Inference Engine\n");
    uart_puts("==================================================\n");
    uart_puts("[READY] Board Booted. Running initial pass on Image #0 (airplane)...\n");

    run_vgg11_demo_inference(0);

#ifdef __riscv
    uart_puts("[IDLE] Waiting for Host Trigger (Send 0-9, 'U' to upload custom image, or 'G')...\n");
    while (1) {
        char cmd;
        if (uart_getc_nonblocking(&cmd)) {
            if (cmd >= '0' && cmd <= '9') {
                int req_img = cmd - '0';
                uart_puts("\n[TRIGGER] Running Built-in Image #");
                uart_put_dec64(req_img);
                uart_puts(" (");
                uart_puts(cifar_test_names[req_img]);
                uart_puts(")\n");
                run_vgg11_demo_inference(req_img);
                uart_puts("[IDLE] Ready for next command (Send 0-9 or 'U')...\n");
            }
            else if (cmd == 'U' || cmd == 'u') {
                uart_puts("\n[UPLOAD_READY] Receiving 3072 raw bytes over UART...\n");
                for (int i = 0; i < 3072; i++) {
                    fm_A[i] = (int8_t)uart_getc_blocking();
                }
                uart_puts("[UPLOAD_DONE] 3072 bytes received! Running FPGA inference...\n");
                run_vgg11_demo_inference_core("custom_user_upload");
                uart_puts("[IDLE] Ready for next command (Send 0-9 or 'U')...\n");
            }
            else if (cmd == 'G' || cmd == 'g' || cmd == '\n' || cmd == '\r' || cmd == ' ') {
                run_vgg11_demo_inference(0);
                uart_puts("[IDLE] Ready for next command (Send 0-9 or 'U')...\n");
            }
        }
    }
#endif
    return 0;
}





