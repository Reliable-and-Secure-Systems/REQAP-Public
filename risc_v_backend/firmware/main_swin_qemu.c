#include <stdio.h>
#include <stdint.h>
#include "inference_ops.h"

int main() {
    printf("==========================================\n");
    printf(" SWIN TRANSFORMER RISC-V KERNEL TEST \n");
    printf("==========================================\n");

    /* 1. TEST Integer LayerNorm */
    printf("\n--- Test 1: LayerNorm ---\n");
    int8_t ln_input[4] = {10, -5, 20, 0};
    int8_t ln_output[4];
    int32_t ln_weight[4] = {256, 256, 256, 256}; // Base scale = 1.0 (256/256)
    int32_t ln_bias[4] = {0, 0, 0, 0};
    
    swar_layer_norm_16bit(ln_input, ln_output, 4, ln_weight, ln_bias);
    
    printf("Input:  ");
    for(int i=0; i<4; i++) printf("%d ", ln_input[i]);
    printf("\nOutput: ");
    for(int i=0; i<4; i++) printf("%d ", ln_output[i]);
    printf("\n");

    /* 2. TEST Integer GELU */
    printf("\n--- Test 2: GELU ---\n");
    int8_t gelu_input[5] = {-100, -10, 0, 10, 100};
    int8_t gelu_output[5];
    
    swar_gelu_approx_16bit(gelu_input, gelu_output, 5);
    
    printf("Input:  ");
    for(int i=0; i<5; i++) printf("%d ", gelu_input[i]);
    printf("\nOutput: ");
    for(int i=0; i<5; i++) printf("%d ", gelu_output[i]);
    printf("\n");

    /* 3. TEST Integer Softmax */
    printf("\n--- Test 3: Softmax ---\n");
    // Softmax probabilities should sum to 127
    int8_t sm_input[4] = {10, 5, 0, -10};
    int8_t sm_output[4];
    
    swar_softmax_approx_16bit(sm_input, sm_output, 4);
    
    printf("Input:  ");
    for(int i=0; i<4; i++) printf("%d ", sm_input[i]);
    printf("\nOutput: ");
    for(int i=0; i<4; i++) printf("%d ", sm_output[i]);
    printf("\n");

    /* 4. TEST Window Attention */
    printf("\n--- Test 4: Window Attention (2x2 window) ---\n");
    // seq_len = 4, head_dim = 2
    int8_t q[8] = {10, 10,   5, 5,   0, 0,  -5, -5};  
    int8_t k[8] = {10, 10,   5, 5,   0, 0,  -5, -5};
    int8_t v[8] = { 1,  1,   2, 2,   3, 3,   4,  4};
    int32_t rel_pos_bias[16] = {0}; // 4x4 matrix initialized to 0
    int8_t wa_output[8];
    
    swar_window_attention_8bit(q, k, v, wa_output, rel_pos_bias, 2, 2, 0.01f, 1.0f);
    
    printf("Output Matrix (Q@K -> Softmax -> @V):\n");
    for(int i=0; i<4; i++) {
        printf("Seq %d: %d %d\n", i, wa_output[i*2], wa_output[i*2+1]);
    }
    
    printf("\n==========================================\n");
    printf(" ALL RISC-V KERNEL TESTS EXECUTED!\n");
    printf("==========================================\n");
    return 0;
}
