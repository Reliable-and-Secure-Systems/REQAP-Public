/*
 * uart_test.c  –  Minimal UART I/O test for QEMU virt (rv32im)
 * Linked with entry.s (which jumps to _start in libgloss/semihost)
 */

#include <stdint.h>

/* QEMU virt 16550-compatible UART */
#define UART_BASE  0x10000000UL
#define UART_THR   (*(volatile uint8_t *)(UART_BASE + 0))
#define UART_LSR   (*(volatile uint8_t *)(UART_BASE + 5))
#define UART_LSR_THRE 0x20

static void uart_putc(char c) {
    while (!(UART_LSR & UART_LSR_THRE));
    UART_THR = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

static void uart_put_uint32(uint32_t v) {
    char buf[11];
    int i = 10;
    buf[i] = '\0';
    if (v == 0) { buf[--i] = '0'; }
    else { while (v) { buf[--i] = '0' + (v % 10); v /= 10; } }
    uart_puts(&buf[i]);
}

/* ------------------------------------------------------------------
 * entry_point — called by startup.s after BSS clear
 * ------------------------------------------------------------------ */
extern char __bss_start;
extern char __BSS_END__;

void entry_point(void) {
    /* Clear BSS */
    char *dst = &__bss_start;
    char *end = &__BSS_END__;
    while (dst < end) *dst++ = 0;

    uart_puts("=== UART TEST START ===\n");
    uart_puts("Hello from RISC-V QEMU!\n");
    uart_puts("42 = ");
    uart_put_uint32(42);
    uart_puts("\n");
    uart_puts("=== UART TEST END ===\n");
}

int main(void) {
    /* entry_point called by startup.s; main is also called by _start */
    return 0;
}
