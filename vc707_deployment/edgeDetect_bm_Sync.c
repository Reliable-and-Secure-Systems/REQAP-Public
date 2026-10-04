
//#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "sun.h"
#include "encoding.h"
#include <stddef.h>
//#include "sim_data.h"
//#include "edges_output.h"

#define CLINT_BASE  0x02000000UL
#define CLINT_MTIME (*(volatile uint64_t *)(CLINT_BASE + 0xBFF8))
#define TIMEBASE_HZ 100000UL   // from your .dts; find it in the fpga/src-generated directory, or in sims/verilator/generated-src directory

// Read the cycle counter (machine mode)
static inline uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

// To reset the execution 
void software_reset(void) {
    // Jump to reset vector (address 0x10000 for Chipyard bootrom)
    void (*reset_vector)(void) = (void (*)(void))0x10000;
    reset_vector();
}

//custom millisecond delay function
static void delay_ms(uint64_t ms) {
    uint64_t ticks = (TIMEBASE_HZ * ms) / 1000;  // convert ms to mtime ticks
    uint64_t end   = CLINT_MTIME + ticks;
    while (CLINT_MTIME < end);
}

// Read instructions retired
static inline uint64_t read_instret(void) {
    uint64_t instret;
    asm volatile ("csrr %0, minstret" : "=r"(instret));
    return instret;
}

//measuring CPU frequency, i.e. if 50Mhz or not?
static inline uint64_t measure_cpu_freq(void) {
    uint64_t c0 = read_cycles();
    uint64_t t0 = CLINT_MTIME;

    // Wait for 1 full second worth of mtime ticks (50000 ticks = 1 sec)
    while ((CLINT_MTIME - t0) < TIMEBASE_HZ);

    uint64_t cycles_per_second = read_cycles() - c0;
    //uart_puts("Frequency: "); uart_print_dec(cycles_per_second);

    return cycles_per_second;

    // cycles_per_second IS your CPU frequency in Hz
    // e.g. if it prints ~50000000 then you're running at 50 MHz
}

// Returns elapsed time in milliseconds
static inline uint64_t mtime_now(void) {
    return CLINT_MTIME; // Current time in mtime ticks
}

static inline uint64_t mtime_elapsed_ms(uint64_t start) {
    return ((CLINT_MTIME - start) * 1000) / TIMEBASE_HZ; // Elapsed time in ms since start
}

static inline uint64_t cycles_elapsed(uint64_t start) {
    return read_cycles() - start;  // Elapsed cycles since start
}


#define NPIX   (WIDTH * HEIGHT)
// extern uint8_t image_data[];
// extern uint8_t image_data_end[];
// extern uint64_t image_data_size;

#define WIDTH  2100
#define HEIGHT 2034
// Statische Arrays (liegen im DRAM bzw. im simulierten RAM)
static uint8_t gray[NPIX];
static uint8_t edges[NPIX];
static uint8_t overlay[NPIX * 3];   // RGB Overlay
//static uint8_t edges_export[NPIX];
//uint8_t edges_output[5][5];

#define UART0 0x64000000
#define UART_TXDATA (*(volatile uint32_t *)(UART0))
#define UART_RXDATA  (*(volatile uint32_t *)(UART0 + 0x04))  
#define UART_TXCTRL (*(volatile uint32_t *)(UART0 + 0x08))
#define UART_RXCTRL  (*(volatile uint32_t *)(UART0 + 0x0C))  // NOT set by bootrom

static inline void uart_send_byte(uint8_t b) {
    while (UART_TXDATA & 0x80000000);  // wait if TX FIFO full
    UART_TXDATA = b;
}

// receive byte on UART
static inline uint8_t uart_recv_byte(void) {
    uint32_t rxval;
    do {
        rxval = UART_RXDATA;
    } while (rxval & 0x80000000);      // bit 31 = empty, same logic as TX full
    return (uint8_t)(rxval & 0xFF);
}

// static void uart_wait_for_start(void) {
//     const char prompt[] = "Press ENTER to start...\n";
//     for (int i = 0; i < sizeof(prompt)-1; i++)
//         uart_send_byte(prompt[i]);

//     uart_recv_byte();   // blocks here until any key is pressed

//     const char ack[] = "Starting...\n";
//     for (int i = 0; i < sizeof(ack)-1; i++)
//         uart_send_byte(ack[i]);
// }

//Waiting for Python to send start command to sync with LMG611
static void uart_wait_for_start(void) {
    uint8_t c;
    do {
        c = uart_recv_byte();
    } while (c != 'S');   // wait specifically for 'S' start signal
}

//Telling Python I am done with executing the code
static void uart_send_done(void) {
    uart_send_byte('D');  // signal Python that algorithm is done
}

static void uart_send_marker(void) {
    const char marker[] = "START_EDGES\n";
    for (int i = 0; i < sizeof(marker)-1; i++)
        uart_send_byte(marker[i]);
}

static void uart_send_image(uint8_t *img, int size) {
    for (int i = 0; i < size; i++)
        uart_send_byte(img[i]);
}

//write_csr(mstatus, read_csr(mstatus) | (3 << 13)); // set FS bits in mstatus
static inline int iabs(int x) {
    return (x < 0) ? -x : x;
}
void uart_putc(char c) {
    if (c == '\n') {
        // send carriage return first
        while (UART_TXDATA & 0x80000000);  
        UART_TXDATA = '\r';
    }
    // send actual character
    while (UART_TXDATA & 0x80000000);  
    UART_TXDATA = c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

void uart_print_hex(uint64_t value) {
    char hex[17];
    hex[16] = '\0';
    for (int i = 15; i >= 0; i--) {
        uint8_t digit = value & 0xF;
        if (digit < 10)
            hex[i] = '0' + digit;
        else
            hex[i] = 'A' + (digit - 10);
        value >>= 4;
    }
    uart_puts(hex);
}

void uart_print_dec(uint64_t v) {
    char buf[20];
    int i = 0;

    if (v == 0) {
        uart_putc('0');
        return;
    }

    while (v) {
        buf[i++] = '0' + (v % 10);
        v /= 10;
    }

    // Print digits in reverse
    while (i--) uart_putc(buf[i]);
}

void uart_print_float(float f, int precision) {
    int int_part = (int)f;
    float frac_part = f - int_part;
    if (f < 0) {
        uart_putc('-');
        int_part = -int_part;
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

int main() {

    UART_RXCTRL = 0x1;
    uint64_t cpu_freq;
    cpu_freq = measure_cpu_freq();


    uart_puts("CPU Frequency = ");

    uart_print_dec(cpu_freq);

    uart_puts(" Hz\n");

    for (int run=0;run<5;run++){

        // Board is ready, tell Python
        uart_send_byte('R');

        uart_wait_for_start();

        //delay_ms(1000);          // wait 1 second
        uint64_t start, end, elapsed;

        uart_send_byte('G');            // "Go" — Python starts energy NOW
        delay_ms(10);          // wait 0.01 second

        //start = read_cycles();
        //uint64_t t0 = mtime_now();


    

        const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        const int Gy[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
        };

        //----------------------------------------------------------------------
        // 3. Sobel-Kantenberechnung
        //----------------------------------------------------------------------


    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {

            int sumX = 0;
            int sumY = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = Sun[(y + ky) * WIDTH + (x + kx)];
                    sumX += pixel * Gx[ky + 1][kx + 1];
                    sumY += pixel * Gy[ky + 1][kx + 1];
                }
            }

            int magnitude = iabs(sumX) + iabs(sumY);
            if (magnitude > 255) magnitude = 255;

            edges[y * WIDTH + x] = (uint8_t)magnitude;
        }
    }

        //----------------------------------------------------------------------
        // 4. Overlay-Bild erzeugen (reines RGB-Array)
        //----------------------------------------------------------------------
        for (int i = 0; i < NPIX; i++) {
            //uint8_t v = gray[i];
            uint8_t v = (uint8_t)Sun[i];

            overlay[3*i + 0] = v;
            overlay[3*i + 1] = v;
            overlay[3*i + 2] = v;

            if (edges[i] > 100) {
                overlay[3*i + 0] = 255; // rot
                overlay[3*i + 1] = 0;
                overlay[3*i + 2] = 0;
            }
        }

        //uint64_t elapsed_ms = mtime_elapsed_ms(t0);
        //uart_puts("Wall clock time: ");
        //uart_print_dec(elapsed_ms);
        //uart_puts(" ms\n");

        //end = read_cycles();
        //elapsed = end - start;

        uart_send_done();            // tell Python to stop the power meter

        //software_reset();

    }

    while(1); //hang forever!


    
}