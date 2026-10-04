/*
 * swar_mac.c
 * ----------
 * SIMD Within A Register (SWAR) MAC kernels for RISC-V.
 *
 * Provides optimized integer MAC routines for MQF Safe-FFD packed weights:
 * - swar_mac_8bit    : 8-bit scalar MAC
 * - swar_mac_4bit_d4 : 4-bit SWAR (4 MACs / instruction)
 * - swar_mac_general : Dynamic precision SWAR via pos/mask LUTs
 */

#include "swar_mac.h"


/* ═══════════════════════════════════════════════════════════════════════════
 * Type 1: swar_mac_8bit
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Covers all layers where MQF assigned 8-bit (dominant_bits=8, max_d=1).
 * In the generated header: LAYER_X_MAX_D == 1
 *
 * Each uint32_t word encodes one signed 8-bit weight in bits [7:0].
 * The upper byte (bits [15:8]) is unused (zero or padding).
 *
 * RISC-V assembly note:
 *   Each iteration is: LHU (load), ANDI (mask to 8b), MUL, ADD
 *   The compiler will emit standard RV32I instructions — no M extension
 *   needed for the mask/shift steps; MUL needs RV32IM.
 *
 * Cycle estimate (RV32IM, in-order): ~4 cycles/iteration
 */
int32_t swar_mac_8bit(const uint32_t *weights,
                      const int8_t   *activations,
                      int             n_words)
{
    int32_t acc = 0;
    int j;

    for (j = 0; j < n_words; j++) {
        // Extract 8-bit signed weight. Upper byte is implicitly ignored.
        int8_t  w = (int8_t)(weights[j] & 0xFF);
        int8_t  a = activations[j];
        acc += (int32_t)w * (int32_t)a;
    }

    return acc;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Type 2: swar_mac_4bit_d4  (the performance-critical hot path)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Covers all layers where MQF assigned 4-bit (dominant_bits=4, max_d=4).
 * In the generated header: LAYER_X_MAX_D == 4, pos={0,4,8,12}, mask=0xF
 *
 * Packing layout of each uint32_t word:
 *
 *   bit 15 ... 12  |  bit 11 ... 8  |  bit 7 ... 4  |  bit 3 ... 0
 *   ───────────────┼────────────────┼───────────────┼──────────────
 *   weight[j*4+3]  |  weight[j*4+2] | weight[j*4+1] | weight[j*4+0]
 *
 * Each 4-bit field is sign-extended before multiplication.
 * Sign extension rule: if bit 3 is set, the value is negative.
 *   val in {0..7}  → positive as-is
 *   val in {8..15} → subtract 16 to get {-8..-1}
 *
 * The four multiplies per word are INDEPENDENT — the compiler (or a
 * future RV32P/Zp extension) can issue them in parallel.
 *
 * Cycle estimate (RV32IM, in-order): ~12-16 cycles per word = 3-4 cycles/MAC
 * vs. 4 cycles/MAC for scalar 8-bit → ~25% improvement per useful bit.
 */
int32_t swar_mac_4bit_d4(const uint32_t *weights,
                          const int8_t   *activations,
                          int             n_words)
{
    int32_t acc = 0;
    int j;

    for (j = 0; j < n_words; j++) {
        uint32_t word = weights[j];

        /* Extract four 4-bit fields.
         * Mask = 0xF, offsets = 0, 4, 8, 12 */
        uint32_t raw0 = (uint32_t)( word        & 0xFU);
        uint32_t raw1 = (uint32_t)((word >>  4) & 0xFU);
        uint32_t raw2 = (uint32_t)((word >>  8) & 0xFU);
        uint32_t raw3 = (uint32_t)((word >> 12) & 0xFU);

        // Sign-extend 4-bit unsigned -> int32
        int32_t w0 = swar_sign_extend(raw0, 4);
        int32_t w1 = swar_sign_extend(raw1, 4);
        int32_t w2 = swar_sign_extend(raw2, 4);
        int32_t w3 = swar_sign_extend(raw3, 4);

        // Scale compensation for 8-bit dynamic range (max=7)
        w0 = (w0 * 127) / 7;
        w1 = (w1 * 127) / 7;
        w2 = (w2 * 127) / 7;
        w3 = (w3 * 127) / 7;

        // Load corresponding activations
        int32_t a0 = (int32_t)activations[j * 4 + 0];
        int32_t a1 = (int32_t)activations[j * 4 + 1];
        int32_t a2 = (int32_t)activations[j * 4 + 2];
        int32_t a3 = (int32_t)activations[j * 4 + 3];

        /* Four independent MACs — accumulate into one 32-bit register */
        acc += w0 * a0;
        acc += w1 * a1;
        acc += w2 * a2;
        acc += w3 * a3;
    }

    return acc;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Type 3: swar_mac_general
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Covers downsample layers and any heterogeneous layer.
 * In the generated header: any layer with max_d > 1 that is not pure 4-bit.
 *
 * pos[] and mask[] come directly from the generated header, e.g.:
 *
 *   swar_mac_general(
 *       layer2_0_downsample_0_weights,
 *       activations,
 *       layer2_0_downsample_0_pos,   // {0, 4, 8, 12} or similar
 *       layer2_0_downsample_0_mask,  // {0xF, 0xF, 0xF, 0xF} or similar
 *       4,    // w_bits_per_field
 *       5,    // d  (max_d from header)
 *       53    // n_words (N_WORDS from header)
 *   );
 *
 * This function is correct for any d and any bit-width.
 * For production code, prefer swar_mac_4bit_d4 for the hot path (d=4 layers).
 *
 * Cycle estimate: ~(d * 5 + 4) cycles per word  (RV32IM, in-order)
 */

static const uint8_t popcount_table[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

int32_t swar_mac_general(const uint32_t *weights,
                          const int8_t   *activations,
                          const uint8_t  *pos,
                          const uint8_t  *mask,
                          const uint16_t *slots,
                          int             d,
                          int             n_words)
{
    int32_t acc = 0;
    int j, k;

    for (j = 0; j < n_words; j++) {
        uint32_t word = weights[j];

        for (k = 0; k < d; k++) {
            int idx = j * d + k;

            /* Determine actual bit-width of the field from its mask dynamically */
            int bits = popcount_table[mask[idx]];

            // Skip unused fields (padding)
            if (bits == 0) continue;

            /* Extract field k from the packed word */
            uint32_t raw = ((uint32_t)word >> pos[idx]) & mask[idx];

            // Sign-extend dynamic width -> int32
            int32_t w = swar_sign_extend(raw, bits);

            // Scale compensation relative to 8-bit range
            if (bits > 1 && bits < 8) {
                int32_t max_int = (1 << (bits - 1)) - 1;
                w = (w * 127) / max_int;
            }

            // Indirect memory fetch for scrambled activations
            int32_t a = (int32_t)activations[slots[idx]];

            acc += w * a;
        }
    }

    return acc;
}

