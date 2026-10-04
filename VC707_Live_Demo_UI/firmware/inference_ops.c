
#include "inference_ops.h"
// Disable semihosting I/O for bare-metal
#ifdef __riscv
#define printf(...) ((void)0)
  #define fprintf(...) ((void)0)

#else
  #include <stdio.h>
#endif

/* 1. Re-Quantization */
int8_t requantize_and_clamp(int32_t accumulator, float scale_factor, int32_t bias) {
    // Shift by bias
    float val = (float)(accumulator + bias);
    // Dynamic scale projection
    val = val * scale_factor;
    
    // Round to nearest integer
    int32_t rounded = (int32_t)(val >= 0 ? val + 0.5f : val - 0.5f);
    
    // Clamp to int8_t range (-128 to 127)
    if (rounded > 127) rounded = 127;
    if (rounded < -128) rounded = -128;
    
    return (int8_t)rounded;
}

int8_t requantize_and_clamp_dynamic(int32_t accumulator, float scale_factor, int32_t bias, uint8_t bit_width) {
    float val = (float)(accumulator + bias) * scale_factor;
    int32_t rounded = (int32_t)(val >= 0 ? val + 0.5f : val - 0.5f);
    
    int32_t max_val = (1 << (bit_width - 1)) - 1;
    int32_t min_val = -(1 << (bit_width - 1));
    
    if (rounded > max_val) rounded = max_val;
    if (rounded < min_val) rounded = min_val;
    
    return (int8_t)rounded;
}

/* 2. ReLU */
void apply_relu(int8_t *activations, int length) {
#ifdef BASELINE_MODE
    for (int i = 0; i < length; i++) {
        if (activations[i] < 0) {
            activations[i] = 0;
        }
    }
#endif
}

/* 3. Max Pool 2x2 (Stride 2) */
void max_pool_2d(const int8_t *input, int8_t *output, 
                 int in_h, int in_w, int channels) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                // Find indices of the 2x2 window
                int i00 = c * (in_h * in_w) + (h * 2) * in_w + (w * 2);
                int i01 = i00 + 1;
                int i10 = i00 + in_w;
                int i11 = i10 + 1;
                
                // Get max
                uint8_t max_val = (uint8_t)input[i00];
                if ((uint8_t)input[i01] > max_val) max_val = (uint8_t)input[i01];
                if ((uint8_t)input[i10] > max_val) max_val = (uint8_t)input[i10];
                if ((uint8_t)input[i11] > max_val) max_val = (uint8_t)input[i11];
                
                output[c * (out_h * out_w) + h * out_w + w] = (int8_t)max_val;
            }
        }
    }
}

void max_pool_3x3_s2_p1(const int8_t *input, int8_t *output, int in_h, int in_w, int channels, int out_h, int out_w) {
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                uint8_t max_val = 0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh * 2 - 1 + kh;
                        int iw = ow * 2 - 1 + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            uint8_t in_val = (uint8_t)input[c * (in_h * in_w) + ih * in_w + iw];
                            if (in_val > max_val) max_val = in_val;
                        } else {
                            if (0 > max_val) max_val = 0;
                        }
                    }
                }
                output[c * (out_h * out_w) + oh * out_w + ow] = (int8_t)max_val;
            }
        }
    }
}

#include <stdlib.h>

/* Hardware Profiler Metric */
uint64_t total_memory_loads = 0;

/* 4. Layer Execution Wrappers (Baseline 8-bit / Unpacked MQF) */
#ifndef __riscv
#include <omp.h>
#endif

void run_conv2d_8bit(int is_signed_input, const int8_t *input, int8_t *output,
                     const int8_t *weights,
                     int in_h, int in_w, int in_ch,
                     int out_h, int out_w, int out_ch,
                     int k_h, int k_w, int pad, int stride,
                     const float *i_scale, const float *w_scale, const float *o_scale, const float *bias,
                     const uint8_t *out_act_bits) 
{
    int K = in_ch * k_h * k_w;
    static int32_t act_window[9216]; // Max K for VGG is 512*3*3 = 4608

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            // Extract 3D window into 1D buffer ONCE for this pixel (oh, ow)
            int idx = 0;
            for (int ic = 0; ic < in_ch; ic++) {
                int ch_offset = ic * (in_h * in_w);
                for (int kh = 0; kh < k_h; kh++) {
                    int ih = oh * stride - pad + kh;
                    if (ih >= 0 && ih < in_h) {
                        int h_offset = ch_offset + ih * in_w;
                        for (int kw = 0; kw < k_w; kw++) {
                            int iw = ow * stride - pad + kw;
                            if (iw >= 0 && iw < in_w) {
                                act_window[idx++] = is_signed_input ? (int8_t)input[h_offset + iw] : (uint8_t)input[h_offset + iw];
                            } else {
                                act_window[idx++] = 0;
                            }
                        }
                    } else {
                        for (int kw = 0; kw < k_w; kw++) {
                            act_window[idx++] = 0;
                        }
                    }
                }
            }

            // Now loop over output channels
            for (int oc = 0; oc < out_ch; oc++) {
                const int8_t *filter_weights = &weights[oc * K];
                float ch_bias = bias[oc];
                
                float float_acc = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    int32_t ic_acc = 0;
                    for (int kh = 0; kh < k_h; kh++) {
                        for (int kw = 0; kw < k_w; kw++) {
                            int f_idx = ic * (k_h * k_w) + kh * k_w + kw;
                            ic_acc += (int32_t)filter_weights[f_idx] * (int32_t)act_window[f_idx];
                        }
                    }
                    float_acc += (float)ic_acc * i_scale[ic];
                }
                
                total_memory_loads += K;

                float final_val = ((float_acc * w_scale[oc]) + ch_bias) / o_scale[oc];
                
                int32_t rounded = (int32_t)(final_val >= 0 ? final_val + 0.5f : final_val - 0.5f);

                /* Skip clamping for classifier output (10‑class logits) */
                if (out_ch != 10) {
                    if (out_act_bits != NULL) {
                        int32_t max_val = 127;
                        int32_t min_val = 0;
                        if (rounded > max_val) rounded = max_val;
                        if (rounded < min_val) rounded = min_val;
                    } else {
                        if (rounded > 127) rounded = 127;
                        if (rounded < -128) rounded = -128;
                    }
                }
                
                output[oc * (out_h * out_w) + oh * out_w + ow] = (int8_t)rounded;
            }
        }
    }
}

void run_linear_8bit(const int8_t *input, int8_t *output,
                     const int8_t *weights,
                     int in_features, int out_features,
                     const float *i_scale, const float *w_scale, const float *o_scale, const float *bias,
                     const uint8_t *out_act_bits) 
{
    for (int oc = 0; oc < out_features; oc++) {
        const int8_t *filter_weights = &weights[oc * in_features];
        float ch_bias = bias[oc];
        
        float float_acc = 0.0f;
        for (int ic = 0; ic < in_features; ic++) {
            int32_t ic_acc = (int32_t)filter_weights[ic] * (int32_t)((uint8_t)input[ic]);
            float_acc += (float)ic_acc * i_scale[ic];
        }
        

        total_memory_loads += in_features;

        float final_val = ((float_acc * w_scale[oc]) + ch_bias) / o_scale[oc];
        
        if (out_features == 10) {
            // printf("Logit %d: float_acc=%f, w_scale=%f, bias=%f, o_scale=%f -> final_val=%f\n", oc, float_acc, w_scale[oc], ch_bias, o_scale[oc], final_val);
        }
        
        int32_t rounded = (int32_t)(final_val >= 0 ? final_val + 0.5f : final_val - 0.5f);

        if (out_features != 10) {
            if (out_features == 10) {
                // No clamping for classifier logits
            } else if (out_act_bits != NULL) {
                int32_t max_val = 127;
                int32_t min_val = 0;
                if (rounded > max_val) rounded = max_val;
                if (rounded < min_val) rounded = min_val;
            } else {
                if (rounded > 127) rounded = 127;
                if (rounded < -128) rounded = -128;
            }
        }
        
        output[oc] = rounded;
    }
}

/* 5. Layer Execution Wrappers (Heterogeneous SWAR) */



void global_average_pool_2d(const int8_t *input, int8_t *output,
                            int h, int w, int channels)
{
    int spatial = h * w;
    for (int c = 0; c < channels; c++) {
        int32_t sum = 0;
        for (int i = 0; i < spatial; i++) {
            sum += input[c * spatial + i];
        }
        output[c] = (int8_t)(sum / spatial);
    }
}

void add_tensors(int8_t *dest, const int8_t *src, int size)
{
    for (int i = 0; i < size; i++) {
        int32_t val = dest[i] + src[i];
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        dest[i] = (int8_t)val;
    }
}



void max_pool_3x3_s2_p0(const int8_t *input, int8_t *output, int in_h, int in_w, int channels, int out_h, int out_w) {
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                uint8_t max_val = 0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int8_t val = input[c * (in_h * in_w) + ih * in_w + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                output[c * (out_h * out_w) + oh * out_w + ow] = (int8_t)max_val;
            }
        }
    }
}

/* =====================================================================
 * SWIN TRANSFORMER SPECIFIC OPERATIONS (HARDWARE FRIENDLY 16-BIT)
 * ===================================================================== */

/* Integer LayerNorm */
void swar_layer_norm_16bit(int8_t *input, int8_t *output, int length, int32_t *weight, int32_t *bias) {
    int32_t sum = 0;
    for (int i = 0; i < length; i++) {
        sum += input[i];
    }
    int32_t mean = sum / length;
    
    int32_t var_sum = 0;
    for (int i = 0; i < length; i++) {
        int32_t diff = input[i] - mean;
        var_sum += diff * diff;
    }
    int32_t variance = var_sum / length;
    
    // Fast Integer Square Root (Standard algorithm)
    int32_t res = 0;
    int32_t bit = 1 << 30; // The second-to-top bit is set
    while (bit > variance) bit >>= 2;
    while (bit != 0) {
        if (variance >= res + bit) {
            variance -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    if (res == 0) res = 1; // Prevent div-by-zero
    
    for (int i = 0; i < length; i++) {
        int32_t diff = input[i] - mean;
        // Fixed-point scaling (Q8)
        int32_t normalized = (diff * 256) / res; 
        
        int32_t val = (normalized * weight[i]) / 256 + bias[i];
        
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[i] = (int8_t)val;
    }
}

/* Integer Softmax (Power of 8 expansion) */
void swar_softmax_approx_16bit(int8_t *input, int8_t *output, int length) {
    uint8_t max_val = 0;
    for (int i = 0; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    int32_t sum_exp = 0;
    // VLA array allocation is supported in C99, but we can just malloc or assume bounded length.
    // For Swin Window attention, length is 49 (7x7 window). Using fixed size buffer for safety:
    int32_t exp_vals[256]; 
    
    for (int i = 0; i < length; i++) {
        int32_t shifted = input[i] - max_val;
        if (shifted < -8) shifted = -8; // Clamp minimum
        
        // Base: (1 + x/8) scaled up by 256 for integer precision
        int32_t base = 256 + (shifted * 32); 
        if (base < 0) base = 0;
        
        // Power of 8 via successive squaring (shift down by 8 to maintain scale)
        int32_t b2 = (base * base) >> 8;
        int32_t b4 = (b2 * b2) >> 8;
        int32_t b8 = (b4 * b4) >> 8;
        
        exp_vals[i] = b8;
        sum_exp += b8;
    }
    
    if (sum_exp == 0) sum_exp = 1;
    
    for (int i = 0; i < length; i++) {
        // Output scaled back to 0-127 range (representing probabilities)
        int32_t prob = (exp_vals[i] * 127) / sum_exp;
        output[i] = (int8_t)prob;
    }
}

/* Integer GELU (Piecewise Sigmoid) */
void swar_gelu_approx_16bit(int8_t *input, int8_t *output, int length) {
    for (int i = 0; i < length; i++) {
        int32_t x = input[i];
        // Sigmoid approx: clamp(0.5 + 0.25*x, 0, 1) -> scaled to 256
        int32_t sig = 128 + (x * 64);
        if (sig < 0) sig = 0;
        if (sig > 256) sig = 256;
        
        int32_t val = (x * sig) >> 8;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[i] = (int8_t)val;
    }
}

/* Swin Window Attention Mechanism (Q @ K^T -> Softmax -> @ V) */
void swar_window_attention_8bit(
    const int8_t *q, const int8_t *k, const int8_t *v, 
    int8_t *output, const int32_t *rel_pos_bias,
    int window_size, int head_dim,
    float scale_qk, float scale_v) 
{
    int seq_len = window_size * window_size; // e.g. 7x7 = 49
    
    // Static allocation to preserve thread stack
    // Max seq_len = 49. 49x49 = 2401 bytes
    static int8_t attn_scores[2401]; 
    static int8_t attn_probs[2401];  
    
    // 1. Q @ K^T Matrix Multiplication
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            int32_t acc = 0;
            for (int d = 0; d < head_dim; d++) {
                acc += (int32_t)q[i * head_dim + d] * (int32_t)k[j * head_dim + d];
            }
            
            // Add relative position bias (Crucial for Swin Transformer)
            acc += rel_pos_bias[i * seq_len + j];
            
            // Re-quantize QK scores before Softmax to prevent overflow
            float scaled = (float)acc * scale_qk;
            int32_t rounded = (int32_t)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
            
            // Softmax expects int8 inputs in our mathematical approximation
            if (rounded > 127) rounded = 127;
            if (rounded < -128) rounded = -128;
            
            attn_scores[i * seq_len + j] = (int8_t)rounded;
        }
    }
    
    // 2. Apply Softmax to each row (each row is size seq_len)
    for (int i = 0; i < seq_len; i++) {
        int8_t *row_in = &attn_scores[i * seq_len];
        int8_t *row_out = &attn_probs[i * seq_len];
        
        swar_softmax_approx_16bit(row_in, row_out, seq_len);
    }
    
    // 3. Attention Probs @ V Matrix Multiplication
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            int32_t acc = 0;
            for (int j = 0; j < seq_len; j++) {
                acc += (int32_t)attn_probs[i * seq_len + j] * (int32_t)v[j * head_dim + d];
            }
            
            // Re-quantize final attention output
            float scaled = (float)acc * scale_v;
            int32_t rounded = (int32_t)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
            if (rounded > 127) rounded = 127;
            if (rounded < -128) rounded = -128;
            
            output[i * head_dim + d] = (int8_t)rounded;
        }
    }
}


int current_act_max_val;

void window_partition_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int window_size) {
    int num_windows_h = H / window_size;
    int num_windows_w = W / window_size;
    int out_idx = 0;
    
    for (int wy = 0; wy < num_windows_h; wy++) {
        for (int wx = 0; wx < num_windows_w; wx++) {
            for (int i = 0; i < window_size; i++) {
                for (int j = 0; j < window_size; j++) {
                    int in_h = wy * window_size + i;
                    int in_w = wx * window_size + j;
                    int in_idx = (in_h * W + in_w) * C;
                    
                    for (int c = 0; c < C; c++) {
                        out[out_idx++] = in[in_idx + c];
                    }
                }
            }
        }
    }
}

void window_reverse_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int window_size) {
    int num_windows_h = H / window_size;
    int num_windows_w = W / window_size;
    int in_idx = 0;
    
    for (int wy = 0; wy < num_windows_h; wy++) {
        for (int wx = 0; wx < num_windows_w; wx++) {
            for (int i = 0; i < window_size; i++) {
                for (int j = 0; j < window_size; j++) {
                    int out_h = wy * window_size + i;
                    int out_w = wx * window_size + j;
                    int out_idx = (out_h * W + out_w) * C;
                    
                    for (int c = 0; c < C; c++) {
                        out[out_idx + c] = in[in_idx++];
                    }
                }
            }
        }
    }
}

void cyclic_shift_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int shift_size) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int in_h = (h + shift_size) % H;
            int in_w = (w + shift_size) % W;
            
            int out_idx = (h * W + w) * C;
            int in_idx = (in_h * W + in_w) * C;
            
            for (int c = 0; c < C; c++) {
                out[out_idx + c] = in[in_idx + c];
            }
        }
    }
}

void reverse_cyclic_shift_8bit(const int8_t *in, int8_t *out, int H, int W, int C, int shift_size) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int in_h = (h - shift_size + H) % H;
            int in_w = (w - shift_size + W) % W;
            
            int out_idx = (h * W + w) * C;
            int in_idx = (in_h * W + in_w) * C;
            
            for (int c = 0; c < C; c++) {
                out[out_idx + c] = in[in_idx + c];
            }
        }
    }
}

void patch_merging_8bit(const int8_t *in, int8_t *out, int H, int W, int C) {
    int out_H = H / 2;
    int out_W = W / 2;
    int out_idx = 0;
    
    for (int h = 0; h < out_H; h++) {
        for (int w = 0; w < out_W; w++) {
            int in_idx0 = ((h * 2) * W + (w * 2)) * C;
            for (int c = 0; c < C; c++) out[out_idx++] = in[in_idx0 + c];
            
            int in_idx1 = ((h * 2 + 1) * W + (w * 2)) * C;
            for (int c = 0; c < C; c++) out[out_idx++] = in[in_idx1 + c];
            
            int in_idx2 = ((h * 2) * W + (w * 2 + 1)) * C;
            for (int c = 0; c < C; c++) out[out_idx++] = in[in_idx2 + c];
            
            int in_idx3 = ((h * 2 + 1) * W + (w * 2 + 1)) * C;
            for (int c = 0; c < C; c++) out[out_idx++] = in[in_idx3 + c];
        }
    }
}










void run_classifier(const int8_t *input, int32_t *output,
                     const int8_t *weights,
                     int in_features, int out_features,
                     const float *i_scale, const float *w_scale, const float *o_scale, const float *bias,
                     const uint8_t *out_act_bits) 
{
    for (int oc = 0; oc < out_features; oc++) {
        const int8_t *filter_weights = &weights[oc * in_features];
        float ch_bias = bias[oc];
        
        float float_acc = 0.0f;
        for (int ic = 0; ic < in_features; ic++) {
            int32_t ic_acc = (int32_t)filter_weights[ic] * (int32_t)((uint8_t)input[ic]);
            float_acc += (float)ic_acc * i_scale[ic];
        }
        
        float final_val = ((float_acc * w_scale[oc]) + ch_bias) / o_scale[oc];
        
        // NO CLAMPING, just direct rounding to int32_t
        int32_t rounded = (int32_t)(final_val >= 0 ? final_val + 0.5f : final_val - 0.5f);
        output[oc] = rounded;
    }
}

void print_layer_range_int32(const char* label, const int32_t* layer, int length) {
#ifndef __riscv
    printf("--- %s ---\n", label);
    int p_len = (length > 10) ? 10 : length;
    for (int i = 0; i < p_len; i++) {
        printf("[%d]: %d\n", i, layer[i]);
    }
#endif
}
