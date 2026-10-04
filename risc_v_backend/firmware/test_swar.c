#include <stdio.h>
#include <stdint.h>
#include "swar_mac.h"

int main() {
    // Just a basic test to make sure it compiles and we can call swar_mac_general
    uint16_t weights[] = {0xFFFF}; // packed data
    int8_t activations[] = {1, 2, 3}; // dummy
    uint8_t pos[] = {0, 5, 10}; // 5-bit packing offsets: 0, 5, 10
    uint8_t mask[] = {0x1F, 0x1F, 0x1F}; // 5-bit mask (31)
    uint16_t slots[] = {0, 1, 2};
    
    int32_t result = swar_mac_general(weights, activations, pos, mask, slots, 3, 1);
    printf("Result of 5-bit SWAR MAC: %d\n", result);
    return 0;
}
