/*
 * baremetal_libc_stubs.c
 * ----------------------
 * Bare-metal libc overrides for VC707 FPGA deployment.
 * Provides custom implementations for standard memory and UART I/O operations
 * (memset, memcpy, printf) to prevent GCC linker errors under -nostdlib.
 */
#include <stdint.h>
#include <stdarg.h>

#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))

void uart_putc(char c) {
    if (c == '\n') {
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\r';
    }
    while (UART_TXDATA & 0x80000000);  
    UART_TXDATA = c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

int puts(const char *s) {
    uart_puts(s);
    uart_putc('\n');
    return 0;
}

int putchar(int c) {
    uart_putc((char)c);
    return c;
}

void uart_print_dec(int64_t v) {
    char buf[20];
    int i = 0;
    
    if (v < 0) {
        uart_putc('-');
        v = -v;
    }
    if (v == 0) {
        uart_putc('0');
        return;
    }
    while (v) {
        buf[i++] = '0' + (v % 10);
        v /= 10;
    }
    while (i--) uart_putc(buf[i]);
}

void uart_print_float(double f, int precision) {
    int64_t int_part = (int64_t)f;
    double frac_part = f - int_part;
    if (f < 0) {
        if (int_part == 0) uart_putc('-');
        frac_part = -frac_part;
    }

    uart_print_dec(int_part);
    uart_putc('.');

    for(int i=0; i<precision; i++) {
        frac_part *= 10;
        int digit = (int)frac_part;
        uart_putc('0' + digit);
        frac_part -= digit;
    }
}

// Bare-metal printf parser
void printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd' || *fmt == 'i') {
                int val = va_arg(args, int);
                uart_print_dec(val);
            } else if (*fmt == 'u') {
                uint32_t val = va_arg(args, uint32_t);
                uart_print_dec(val);
            } else if (*fmt == 'l') {
                fmt++;
                if (*fmt == 'u') {
                    uint64_t val = va_arg(args, uint64_t);
                    uart_print_dec(val);
                } else if (*fmt == 'l' && *(fmt+1) == 'u') {
                    fmt++;
                    uint64_t val = va_arg(args, uint64_t);
                    uart_print_dec(val);
                }
            } else if (*fmt == 's') {
                char *s = va_arg(args, char *);
                uart_puts(s ? s : "(null)");
            } else if (*fmt == 'c') {
                char c = (char)va_arg(args, int);
                uart_putc(c);
            } else if (*fmt == 'f') {
                double val = va_arg(args, double);
                uart_print_float(val, 6);
            } else if (*fmt == '-') {
                // Ignore formatting padding modifiers like %-25s
                while (*fmt >= '0' && *fmt <= '9') fmt++;
                if (*fmt == 's') {
                    char *s = va_arg(args, char *);
                    uart_puts(s ? s : "(null)");
                }
            } else if (*fmt >= '0' && *fmt <= '9') {
                // Ignore padding like %4d
                while (*fmt >= '0' && *fmt <= '9') fmt++;
                if (*fmt == 'd') {
                    int val = va_arg(args, int);
                    uart_print_dec(val);
                }
            } else if (*fmt == '%') {
                uart_putc('%');
            }
        } else {
            uart_putc(*fmt);
        }
        fmt++;
    }
    va_end(args);
}

int abs(int j) { return j < 0 ? -j : j; }
void setvbuf(void *stream, char *buf, int mode, int size) { }

void* _impure_ptr = 0;
char __bss_start = 0;
char __BSS_END__ = 0;

// Memory operations needed for bare-metal
void *memcpy(void *dest, const void *src, uint32_t n) {
    char *d = dest;
    const char *s = src;
    while (n--) *d++ = *s++;
    return dest;
}

void *memset(void *s, int c, uint32_t n) {
    unsigned char *p = s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}
