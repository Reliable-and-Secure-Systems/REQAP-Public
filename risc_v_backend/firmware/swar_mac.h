/*
 * swar_mac.h
 * ==========
 * SWAR (SIMD Within A Register) MAC kernel for RISC-V
 *
 * Provides three MAC functions covering the three vector types that the
 * MQF Safe-FFD packer produces in the generated header files:
 *
 *   Type 1 — d=1, 8-bit weights (no packing)
 *             One weight per uint16_t word, stored in lower 8 bits.
 *             Used by: conv1, layer*.conv1, layer*.conv3, fc
 *
 *   Type 2 — d=4, 4-bit weights (SWAR packed, 4 MACs per word)
 *             Four 4-bit weights packed at bit offsets {0, 4, 8, 12}.
 *             Used by: layer*.conv2 (depth-wise/pointwise bottleneck convs)
 *
 *   Type 3 — d=N, heterogeneous (general case, any d and any bit-widths)
 *             Uses pos[] and mask[] arrays from the generated header.
 *             Used by: downsample layers and any mixed-precision layer.
 *
 * All functions compile cleanly on standard GCC and riscv-none-embed-gcc.
 * No intrinsics, no extensions — pure C99.
 *
 * Compile (PC test):
 *   gcc -std=c99 -O2 -Wall -I generated/ swar_mac.c -c
 *
 * Compile (RISC-V):
 *   riscv-none-embed-gcc -march=rv32i -mabi=ilp32 -O2 -std=c99 swar_mac.c -c
 */

#ifndef SWAR_MAC_H
#define SWAR_MAC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* ── Type 1: d=1, 8-bit weights, no packing ─────────────────────────────────
 *
 * Each word holds one signed 8-bit weight in the lower byte.
 * activations[] is a flat int8_t array, one value per K slot.
 *
 * Parameters:
 *   weights     — packed weight array (uint16_t, lower 8 bits used)
 *   activations — int8_t activation array, length n_words
 *   n_words     — number of K slots (= n_words for d=1)
 *
 * Returns: 32-bit dot-product accumulator
 */
int32_t swar_mac_8bit(const uint32_t *weights,
                      const int8_t   *activations,
                      int             n_words);


/* ── Type 2: d=4, 4-bit weights, fixed SWAR packing ─────────────────────────
 *
 * Each uint16_t word holds FOUR 4-bit signed weights at fixed offsets:
 *   field 0: bits [3:0]   offset=0,  mask=0xF
 *   field 1: bits [7:4]   offset=4,  mask=0xF
 *   field 2: bits [11:8]  offset=8,  mask=0xF
 *   field 3: bits [15:12] offset=12, mask=0xF
 *
 * activations[] is a flat int8_t array, one value per original K slot.
 * The mapping is: word j contains K slots {j*4, j*4+1, j*4+2, j*4+3}.
 *
 * This function is the performance-critical inner loop.
 * The inner loop body is 4 independent multiply-accumulates per iteration,
 * which a RISC-V compiler can pipeline efficiently.
 *
 * Parameters:
 *   weights     — packed weight array (uint16_t), length n_words
 *   activations — int8_t activation array, length n_words * 4
 *   n_words     — number of packed words
 *
 * Returns: 32-bit dot-product accumulator
 */
int32_t swar_mac_4bit_d4(const uint32_t *weights,
                          const int8_t   *activations,
                          int             n_words);


/* ── Type 3: General SWAR MAC (any d, any bit-widths) ───────────────────────
 *
 * Uses the pos[] and mask[] metadata arrays from the generated header file
 * to unpack fields dynamically. Handles any packing depth d and any mix of
 * bit-widths within a word.
 *
 * This is the correct function for downsample layers and any layer where
 * the packing is heterogeneous (different bit-widths in the same word).
 *
 * Internally calls sign_extend() to correctly handle signed weights at
 * any bit-width.
 *
 * Parameters:
 *   weights     — packed weight array (uint16_t), length n_words
 *   activations — int8_t activation array, length n_words * d
 *   pos         — bit offset of each field, length d (from generated header)
 *   mask        — bit mask of each field,   length d (from generated header)
 *   w_bits_per_field — bits per weight field (same for all fields in this call;
 *                      for truly heterogeneous use swar_mac_general_hetero)
 *   d           — number of fields per word
 *   n_words     — number of packed words
 *
 * Returns: 32-bit dot-product accumulator
 */
int32_t swar_mac_general(const uint32_t *weights,
                          const int8_t   *activations,
                          const uint8_t  *pos,
                          const uint8_t  *mask,
                          const uint16_t *slots,
                          int             d,
                          int             n_words);


/* ── Helper: sign-extend a b-bit unsigned value to int32 ────────────────────
 *
 * Example: sign_extend(0xF, 4) = -1
 *          sign_extend(0x7, 4) = +7
 *          sign_extend(0xFF, 8) = -1
 *
 * Exposed as inline so the MAC loops can be inlined by the compiler.
 */
static inline int32_t swar_sign_extend(uint32_t val, int bits)
{
    int shift = 32 - bits;
    return (int32_t)(val << shift) >> shift;
}


#ifdef __cplusplus
}
#endif

#endif /* SWAR_MAC_H */
