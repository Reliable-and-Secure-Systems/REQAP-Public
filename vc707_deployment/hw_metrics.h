#ifndef HW_METRICS_H
#define HW_METRICS_H

#include <stdint.h>

// RV64 CSR access via inline assembly.

static inline uint64_t read_mcycle() {
    uint64_t cycles;
    // Read mcycle CSR
    __asm__ volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

static inline uint64_t read_minstret() {
    uint64_t instret;
    // Read minstret CSR
    __asm__ volatile ("rdinstret %0" : "=r" (instret));
    return instret;
}

#endif // HW_METRICS_H
