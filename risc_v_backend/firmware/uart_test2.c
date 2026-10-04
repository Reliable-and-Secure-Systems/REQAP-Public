/*
 * uart_test2.c — Minimal UART test using only main()
 * Links with entry.s -> _start (libc crt0) -> main()
 */
#include <stdint.h>

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

int main(void) {
    uart_puts("=== UART TEST START ===\n");
    uart_puts("Hello from RISC-V QEMU!\n");
    uart_puts("=== UART TEST END ===\n");

    /* Spin to keep QEMU alive long enough to flush */
    volatile int i;
    for (i = 0; i < 100000; i++);

    return 0;
}
