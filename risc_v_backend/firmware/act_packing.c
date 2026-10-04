#include "act_packing.h"

uint32_t compute_packed_layout(const uint8_t *out_act_bits, uint32_t *offset_table,
                               int channels, int spatial_size) 
{
    uint32_t current_offset = 0;
    for (int c = 0; c < channels; c++) {
        offset_table[c] = current_offset;
        int b = out_act_bits[c];
        if (b == 8) {
            current_offset += spatial_size;
        } else {
            // ceil(spatial_size * b / 8.0)
            current_offset += (spatial_size * b + 7) / 8;
        }
    }
    return current_offset;
}

void pack_feature_map(const int8_t *input, uint8_t *packed_output,
                      const uint8_t *out_act_bits, const uint32_t *offset_table,
                      int channels, int spatial_size) 
{
    // PASS 1: Find the true maximum absolute value in the entire tensor
    int max_val = 1; // Prevent division by zero
    int total_pixels = channels * spatial_size;
    for (int i = 0; i < total_pixels; i++) {
        int v = input[i];
        if (v < 0) v = -v;
        if (v > max_val) max_val = v;
    }
    current_act_max_val = max_val;

    for (int c = 0; c < channels; c++) {
        int b = out_act_bits[c];
        const int8_t *chan_input = &input[c * spatial_size];
        uint8_t *chan_output = &packed_output[offset_table[c]];
        
        if (b == 8) {
            // Fast copy for 8-bit channels
            for (int i = 0; i < spatial_size; i++) {
                chan_output[i] = (uint8_t)chan_input[i];
            }
        } else {
            // Pack b-bit pixels
            uint8_t current_byte = 0;
            int bits_in_byte = 0;
            int byte_idx = 0;
            uint8_t mask = (1 << b) - 1;
            
            for (int i = 0; i < spatial_size; i++) {
                int max_int = (1 << b) - 1; // Unsigned max since activations follow ReLU
                int32_t val = chan_input[i];
                if (val >= 0) {
                    val = (val * max_int + (max_val / 2)) / max_val;
                } else {
                    val = (val * max_int - (max_val / 2)) / max_val;
                }
                uint8_t raw = (uint8_t)val & mask;
                
                int bits_to_pack = b;
                while (bits_to_pack > 0) {
                    int space_in_byte = 8 - bits_in_byte;
                    if (bits_to_pack <= space_in_byte) {
                        current_byte |= (raw << bits_in_byte);
                        bits_in_byte += bits_to_pack;
                        bits_to_pack = 0;
                    } else {
                        current_byte |= (raw << bits_in_byte);
                        chan_output[byte_idx++] = current_byte;
                        raw >>= space_in_byte;
                        bits_to_pack -= space_in_byte;
                        current_byte = 0;
                        bits_in_byte = 0;
                    }
                }
                
                if (bits_in_byte == 8) {
                    chan_output[byte_idx++] = current_byte;
                    current_byte = 0;
                    bits_in_byte = 0;
                }
            }
            if (bits_in_byte > 0) {
                chan_output[byte_idx++] = current_byte;
            }
        }
    }
}
