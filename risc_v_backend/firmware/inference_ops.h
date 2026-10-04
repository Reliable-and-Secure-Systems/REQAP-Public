#ifndef INFERENCE_OPS_H
#define INFERENCE_OPS_H

#include <stdint.h>

/* 1. Re-Quantization: Convert 32-bit MAC accumulator back to 8-bit activation 
 * Uses the scale factor from the MQF JSON config.
 */
int8_t requantize_and_clamp(int32_t accumulator, float scale_factor, int32_t bias);

/* 2. ReLU Activation (In-place) */
void apply_relu(int8_t *activations, int length);

/* 3. Max Pooling 2x2 (Stride 2) */
void max_pool_2d(const int8_t *input, int8_t *output, 
                 int in_h, int in_w, int channels);

void max_pool_3x3_s2_p1(const int8_t *input, int8_t *output, int in_h, int in_w, int channels, int out_h, int out_w);

void global_average_pool_2d(const int8_t *input, int8_t *output,
                            int h, int w, int channels);

/* 3.1. Tensor addition (for ResNet skip connections) */
void add_tensors(int8_t *dest, const int8_t *src, int size);

/* --- Hardware Profiler Metrics --- */
extern uint64_t total_memory_loads;

/* 4. Layer Execution Wrappers (Baseline 8-bit) */
void run_conv2d_8bit(const int8_t *input, int8_t *output,
                     const uint32_t *packed_weights,
                     int in_h, int in_w, int in_ch,
                     int out_h, int out_w, int out_ch,
                     int k_h, int k_w, int pad, int stride,
                     const float *scale, const int32_t *bias);

void run_linear_8bit(const int8_t *input, int8_t *output,
                     const uint32_t *packed_weights,
                     int in_features, int out_features,
                     const float *scale, const int32_t *bias);

/* 5. Layer Execution Wrappers (Packed SWAR 4-bit) */

/* 5. Layer Execution Wrappers (Heterogeneous SWAR) */
void run_conv2d_general(const uint8_t *packed_input, int8_t *output,
                        const uint32_t *packed_weights,
                        const uint8_t *pos, const uint8_t *mask, const uint16_t *slots,
                        const uint8_t *in_act_bits, const uint32_t *in_offset_table,
                        int in_h, int in_w, int in_ch,
                        int out_h, int out_w, int out_ch,
                        int k_h, int k_w, int pad, int stride,
                        const float *scale, const int32_t *bias,
                        int d, int words_per_filter);

void run_linear_general(const uint8_t *packed_input, int8_t *output,
                        const uint32_t *packed_weights,
                        const uint8_t *pos, const uint8_t *mask, const uint16_t *slots,
                        const uint8_t *in_act_bits, const uint32_t *in_offset_table,
                        int in_features, int out_features,
                        const float *scale, const int32_t *bias,
                        int d, int words_per_filter);

void run_depthwise_conv2d_8bit(const int8_t *input, int8_t *output,
                               const uint32_t *packed_weights,
                               int in_h, int in_w, int channels,
                               int out_h, int out_w, 
                               int k_h, int k_w, int pad, int stride,
                               const float *scale, const int32_t *bias);

void run_depthwise_conv2d_general(const uint8_t *packed_input, int8_t *output,
                                  const uint32_t *packed_weights,
                                  const uint8_t *pos, const uint8_t *mask, const uint16_t *slots,
                                  const uint8_t *in_act_bits, const uint32_t *in_offset_table,
                                  int in_h, int in_w, int channels,
                                  int out_h, int out_w, 
                                  int k_h, int k_w, int pad, int stride,
                                  const float *scale, const int32_t *bias,
                                  int d, int words_per_filter);


/* 6. Swin Transformer Specific Operations */
void swar_layer_norm_16bit(int8_t *input, int8_t *output, int length, int32_t *weight, int32_t *bias);
void swar_softmax_approx_16bit(int8_t *input, int8_t *output, int length);
void swar_gelu_approx_16bit(int8_t *input, int8_t *output, int length);
void swar_window_attention_8bit(
    const int8_t *q, const int8_t *k, const int8_t *v, 
    int8_t *output, const int32_t *rel_pos_bias,
    int window_size, int head_dim,
    float scale_qk, float scale_v);

void max_pool_3x3_s2_p0(const int8_t *input, int8_t *output, int in_h, int in_w, int channels, int out_h, int out_w);

void window_partition_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int window_size);
void window_reverse_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int window_size);
void cyclic_shift_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int shift_size);
void reverse_cyclic_shift_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int shift_size);
void patch_merging_8bit(const int8_t *in, int8_t *out, int H, int W, int C);

#endif // INFERENCE_OPS_H
