# Makefile for Physical VC707 FPGA Deployment (Chipyard Rocket RV64)
# =================================================================

CC = riscv64-unknown-elf-gcc
CFLAGS = -march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -I. -I../risc_v_backend -I../risc_v_backend/firmware
LDFLAGS = -nostdlib -T link_uart.ld -lm -lgcc

COMMON_SRCS = ../risc_v_backend/firmware/inference_ops.c ../risc_v_backend/firmware/swar_mac.c

all: swin_vc707 vgg11_packed vgg11_baseline resnet18_packed resnet18_baseline alexnet_packed alexnet_baseline

# --- Boot and Dummy Libc ---
boot1.o: boot1.S
	$(CC) $(CFLAGS) -c boot1.S -o boot1.o

baremetal_libc_stubs.o: baremetal_libc_stubs.c
	$(CC) $(CFLAGS) -c baremetal_libc_stubs.c -o baremetal_libc_stubs.o

# --- Swin Transformer Target ---
# Using the QEMU test file as the main entry point for the kernel tests
main_swin_qemu.o: ../risc_v_backend/firmware/main_swin_qemu.c
	$(CC) $(CFLAGS) -c ../risc_v_backend/firmware/main_swin_qemu.c -o main_swin_qemu.o

inference_ops.o: ../risc_v_backend/firmware/inference_ops.c
	$(CC) $(CFLAGS) -c ../risc_v_backend/firmware/inference_ops.c -o inference_ops.o

swar_mac.o: ../risc_v_backend/firmware/swar_mac.c
	$(CC) $(CFLAGS) -c ../risc_v_backend/firmware/swar_mac.c -o swar_mac.o

swin_vc707: boot1.o baremetal_libc_stubs.o main_swin_qemu.o inference_ops.o swar_mac.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_swin_qemu.o inference_ops.o swar_mac.o -o swin_vc707.elf
	@echo "Built swin_vc707.elf for RV64 VC707 deployment"

# --- VGG11 Targets ---
vgg11_packed: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_VGG11 -c main_vc707_vgg11.c -o main_vc707_vgg11.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/vgg11_packed_weights.cc -o vgg11_packed_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_vgg11.o vgg11_packed_weights.o inference_ops.o swar_mac.o -o vgg11_packed_vc707.elf
	@echo "Built vgg11_packed_vc707.elf for RV64 VC707 deployment"

vgg11_baseline: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_VGG11 -DBASELINE_MODE -c main_vc707_vgg11.c -o main_vc707_vgg11_base.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/vgg11_baseline_weights.cc -o vgg11_baseline_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_vgg11_base.o vgg11_baseline_weights.o inference_ops.o swar_mac.o -o vgg11_baseline_vc707.elf
	@echo "Built vgg11_baseline_vc707.elf for RV64 VC707 deployment"

# --- ResNet18 Targets ---
resnet18_packed: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_RESNET18 -c main_vc707_resnet18.c -o main_vc707_resnet18.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/resnet18_packed_weights.cc -o resnet18_packed_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_resnet18.o resnet18_packed_weights.o inference_ops.o swar_mac.o -o resnet18_packed_vc707.elf
	@echo "Built resnet18_packed_vc707.elf for RV64 VC707 deployment"

resnet18_baseline: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_RESNET18 -DBASELINE_MODE -c main_vc707_resnet18.c -o main_vc707_resnet18_base.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/resnet18_baseline_weights.cc -o resnet18_baseline_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_resnet18_base.o resnet18_baseline_weights.o inference_ops.o swar_mac.o -o resnet18_baseline_vc707.elf
	@echo "Built resnet18_baseline_vc707.elf for RV64 VC707 deployment"

# --- AlexNet Targets ---
alexnet_packed: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_ALEXNET -c main_vc707_alexnet.c -o main_vc707_alexnet.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/alexnet_packed_weights.cc -o alexnet_packed_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_alexnet.o alexnet_packed_weights.o inference_ops.o swar_mac.o -o alexnet_packed_vc707.elf
	@echo "Built alexnet_packed_vc707.elf for RV64 VC707 deployment"

alexnet_baseline: boot1.o baremetal_libc_stubs.o
	$(CC) $(CFLAGS) -DTEST_ALEXNET -DBASELINE_MODE -c main_vc707_alexnet.c -o main_vc707_alexnet_base.o
	$(CC) $(CFLAGS) -c ../risc_v_backend/generated/alexnet_baseline_weights.cc -o alexnet_baseline_weights.o
	$(CC) $(CFLAGS) $(LDFLAGS) boot1.o baremetal_libc_stubs.o main_vc707_alexnet_base.o alexnet_baseline_weights.o inference_ops.o swar_mac.o -o alexnet_baseline_vc707.elf
	@echo "Built alexnet_baseline_vc707.elf for RV64 VC707 deployment"

clean:
	del /Q *.elf *.o 2>nul || rm -f *.elf *.o
