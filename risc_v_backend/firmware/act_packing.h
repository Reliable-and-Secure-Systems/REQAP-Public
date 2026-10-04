#ifndef ACT_PACKING_H
#define ACT_PACKING_H

#include <stdint.h>
#include <stddef.h>

extern int current_act_max_val;

// Compute the layout (offset table and total size) for a heterogeneous packed feature map.
// out_act_bits: array of bit-widths per channel
// offset_table: pre-allocated array of size 'channels' to hold computed byte offsets
// returns: total bytes required for the packed buffer
uint32_t compute_packed_layout(const uint8_t *out_act_bits, uint32_t *offset_table,
                               int channels, int spatial_size);

// Pack a full CHW feature map from 8-bit down to heterogeneous bit-widths.
void pack_feature_map(const int8_t *input, uint8_t *packed_output,
                      const uint8_t *out_act_bits, const uint32_t *offset_table,
                      int channels, int spatial_size);

// Extract a single pixel from the heterogeneous packed feature map.
static inline int8_t extract_packed_pixel(const uint8_t *packed_input,
                                          int channel, int pixel_idx,
                                          const uint8_t *act_bits, 
                                          const uint32_t *offset_table) 
{
    int b = act_bits[channel];
    if (b == 8) {
        return (int8_t)packed_input[offset_table[channel] + pixel_idx];
    }
    
    // For b < 8 (e.g., 2, 4)
    int bit_idx = pixel_idx * b;
    int byte_offset = bit_idx / 8;
    int bit_shift = bit_idx % 8;
    
    uint8_t byte_val = packed_input[offset_table[channel] + byte_offset];
    
    // Extract b bits
    uint8_t mask = (1 << b) - 1;
    uint8_t raw = (byte_val >> bit_shift) & mask;
    
    // Sign extend b bits to 8-bit
    int shift = 8 - b;
    int8_t extended = (int8_t)(raw << shift) >> shift;
    
    // Scale compensation for dynamic range
    int max_int = (1 << b) - 1; // Unsigned max since activations follow ReLU
    if (max_int > 0) {
        if (extended >= 0) {
            extended = (extended * current_act_max_val + (max_int / 2)) / max_int;
        } else {
            extended = (extended * current_act_max_val - (max_int / 2)) / max_int;
        }
    }
    
    return extended;
}

#endif // ACT_PACKING_H
