#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../generated/swin_packed_weights.h"

// Define architecture parameters
#define SEQ_LEN 64
#define EMBED_DIM_0 96
#define HEADS_0 3
#define HEAD_DIM (EMBED_DIM_0 / HEADS_0)

// Helper: Unpack a single weight from the Safe-FFD 16-bit register word
static inline int8_t unpack_weight_16(uint16_t word, uint8_t offset, uint8_t mask, uint8_t bits) {
    uint8_t raw = (word >> offset) & mask;
    // Sign extend based on the bitwidth
    int8_t shift = 8 - bits;
    return (int8_t)(raw << shift) >> shift;
}

// Emulate a standard Multi-Head Self Attention (MHSA) projection using packed SWAR weights
// For Swin-T block 0 (layers_0_blocks_0_attn_qkv)
void mhsa_qkv_projection(float* input, float* qkv_out) {
    // input is [SEQ_LEN][EMBED_DIM_0]
    // qkv_out is [SEQ_LEN][3 * EMBED_DIM_0]
    
    // QKV weights dimension: K = EMBED_DIM_0 (96), P = 3 * EMBED_DIM_0 (288)
    int n_words = LAYERS_0_BLOCKS_0_ATTN_QKV_N_WORDS;
    int words_per_filter = LAYERS_0_BLOCKS_0_ATTN_QKV_WORDS_PER_FILTER;
    
    for (int seq = 0; seq < SEQ_LEN; seq++) {
        for (int p = 0; p < 288; p++) { // Output channel for Q, K, V
            float acc = layers_0_blocks_0_attn_qkv_bias[p];
            float scale = 1.0f; // Scale simplified for bare-metal demo
            
            // Unpack SWAR weights for this filter
            int word_idx = p * words_per_filter;
            int k_idx = 0;
            
            for (int w = 0; w < words_per_filter; w++) {
                uint16_t reg_word = layers_0_blocks_0_attn_qkv_weights[word_idx + w];
                
                // Assuming max_d packing depth of 4 (example)
                // In actual firmware, we iterate up to MAX_D using the pos and mask arrays
                for (int d = 0; d < 4; d++) {
                    if (k_idx >= EMBED_DIM_0) break;
                    
                    int meta_idx = w * 4 + d;
                    uint8_t pos = layers_0_blocks_0_attn_qkv_pos[meta_idx];
                    uint8_t mask = layers_0_blocks_0_attn_qkv_mask[meta_idx];
                    
                    // The number of bits is inferred by the mask
                    uint8_t bits = 0;
                    for (int b = 0; b < 8; b++) {
                        if ((mask >> b) & 1) bits++;
                    }
                    
                    if (bits > 0) {
                        int8_t q_weight = unpack_weight_16(reg_word, pos, mask, bits);
                        acc += input[seq * EMBED_DIM_0 + k_idx] * ((float)q_weight * scale);
                        k_idx++;
                    }
                }
            }
            qkv_out[seq * 288 + p] = acc;
        }
    }
}

// Dummy main function to ensure compilation against the RISC-V toolchain
int main() {
    printf("--- SWIN Transformer (ViT) MQF Bare-Metal Execution ---\n");
    
    // Allocate dummy input and output buffers
    float* input_embeds = (float*)malloc(SEQ_LEN * EMBED_DIM_0 * sizeof(float));
    float* qkv_output = (float*)malloc(SEQ_LEN * 3 * EMBED_DIM_0 * sizeof(float));
    
    // Initialize with dummy data
    for (int i = 0; i < SEQ_LEN * EMBED_DIM_0; i++) {
        input_embeds[i] = 0.5f;
    }
    
    // Execute standard MHSA QKV projection
    mhsa_qkv_projection(input_embeds, qkv_output);
    
    printf("Successfully executed MHSA QKV Projection using 16-bit packed weights!\n");
    printf("Sample QKV output [0]: %f\n", qkv_output[0]);
    
    free(input_embeds);
    free(qkv_output);
    
    return 0;
}
