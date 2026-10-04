CC         = gcc
CXX        = g++

# Supports both Linux (riscv64-unknown-elf-gcc) and Windows (riscv-none-elf-gcc)
RV_CC      ?= $(shell where riscv64-unknown-elf-gcc 2>nul || where riscv-none-elf-gcc 2>nul || which riscv64-unknown-elf-gcc 2>/dev/null || which riscv-none-elf-gcc 2>/dev/null || echo riscv64-unknown-elf-gcc)
RV_CXX     ?= $(shell where riscv64-unknown-elf-g++ 2>nul || where riscv-none-elf-g++ 2>nul || which riscv64-unknown-elf-g++ 2>/dev/null || which riscv-none-elf-g++ 2>/dev/null || echo riscv64-unknown-elf-g++)



CFLAGS     = -O2 -I. -Wall -Wno-unused-function -DBASELINE_MODE
RV_CFLAGS  = -O2 -I. -march=rv64gc -mabi=lp64d -mcmodel=medany -ffreestanding -nostartfiles -T link_uart.ld -DBASELINE_MODE

SRCS_C_PC  = main_vc707_demo.c inference_ops.c
SRCS_C_RV  = main_vc707_demo.c inference_ops.c baremetal_libc_stubs.c
WEIGHTS_CC = vgg11_baseline_unpacked.cc

all: pc riscv

pc: vgg11_pc_demo.exe

riscv: vgg11_rv_demo_vc707.elf

vgg11_baseline_unpacked.o: $(WEIGHTS_CC)
	$(CXX) -O0 -c $(WEIGHTS_CC) -o $@ -std=c++11

vgg11_pc_demo.exe: $(SRCS_C_PC) vgg11_baseline_unpacked.o
	$(CXX) $(CFLAGS) $(SRCS_C_PC) vgg11_baseline_unpacked.o -o $@ -lm

vgg11_rv_weights.o: $(WEIGHTS_CC)
	$(RV_CXX) -O0 -c $(WEIGHTS_CC) -o $@ -march=rv64gc -mabi=lp64d -mcmodel=medany -ffreestanding -std=c++11

vgg11_rv_demo_vc707.elf: $(SRCS_C_RV) boot1.S link_uart.ld vgg11_rv_weights.o
	$(RV_CC) $(RV_CFLAGS) boot1.S $(SRCS_C_RV) vgg11_rv_weights.o -o $@ -lm


clean:
	rm -f *.o *.exe *.elf
