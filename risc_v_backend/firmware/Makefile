# Makefile — SWAR MAC Kernel Build System (all models)
# =====================================================
# Targets:
#   make resnet18  — build and run ResNet-18 test
#   make alexnet   — build and run AlexNet test
#   make vgg11     — build and run VGG-11 test
#   make all       — build and run all three
#   make riscv     — cross-compile ResNet-18 for RISC-V
#   make clean     — remove build artifacts

CC_PC    = gcc
CC_RISCV = riscv-none-elf-gcc

# -x c++ needed because the generated .cc files use C++ extension but are C code
CFLAGS_PC    = -O2 -Wall -I. -I generated/
CFLAGS_RISCV = -O2 -Wall -march=rv32i -mabi=ilp32 -I. -I generated/ \
               -ffunction-sections -fdata-sections

COMMON_SRCS = main_inference.c swar_mac.c

# ── Per-model targets ─────────────────────────────────────────────────────────

resnet18: $(COMMON_SRCS) swar_mac.h generated/resnet18_packed_weights.h \
           generated/resnet18_packed_weights.cc
	$(CC_PC) $(CFLAGS_PC) -std=c99 $(COMMON_SRCS) \
	    generated/resnet18_packed_weights.cc \
	    -DTEST_RESNET18 -o test_resnet18
	@echo ""
	@echo "--- ResNet-18 C output ---"
	./test_resnet18
	@echo ""
	@echo "--- ResNet-18 Python ground truth ---"
	python verify_output.py --model resnet18

alexnet: $(COMMON_SRCS) swar_mac.h generated/alexnet_packed_weights.h \
          generated/alexnet_packed_weights.cc
	$(CC_PC) $(CFLAGS_PC) -std=c99 $(COMMON_SRCS) \
	    generated/alexnet_packed_weights.cc \
	    -DTEST_ALEXNET -o test_alexnet
	@echo ""
	@echo "--- AlexNet C output ---"
	./test_alexnet
	@echo ""
	@echo "--- AlexNet Python ground truth ---"
	python verify_output.py --model alexnet

vgg11: $(COMMON_SRCS) swar_mac.h generated/vgg11_packed_weights.h \
        generated/vgg11_packed_weights.cc
	$(CC_PC) $(CFLAGS_PC) -std=c99 $(COMMON_SRCS) inference_ops.c \
	    generated/vgg11_packed_weights.cc \
	    -DTEST_VGG11 -o test_vgg11
	@echo ""
	@echo "--- VGG-11 Packed MQF C output ---"
	./test_vgg11
	@echo ""
	@echo "--- VGG-11 Python ground truth ---"
	python verify_output.py --model vgg11

vgg11-baseline: $(COMMON_SRCS) swar_mac.h generated/vgg11_baseline_weights.h \
        generated/vgg11_baseline_weights.cc
	$(CC_PC) $(CFLAGS_PC) -std=c99 $(COMMON_SRCS) inference_ops.c \
	    generated/vgg11_baseline_weights.cc \
	    -DTEST_VGG11 -DBASELINE_MODE -o test_vgg11_baseline
	@echo ""
	@echo "--- VGG-11 Baseline 8-bit C output ---"
	./test_vgg11_baseline
	@echo ""
	@echo "--- VGG-11 Python ground truth ---"
	python verify_output.py --model vgg11

vgg11-compare: vgg11 vgg11-baseline
	@echo ""
	@echo "--- Packed MQF Results ---"
	@type test_vgg11
	@echo ""
	@echo "--- Baseline 8-bit Results ---"
	@type test_vgg11_baseline

qemu-vgg11: $(COMMON_SRCS) swar_mac.h generated/vgg11_packed_weights.h \
        generated/vgg11_packed_weights.cc
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    main_inference.c swar_mac.c inference_ops.c generated/vgg11_packed_weights.cc \
	    -DTEST_VGG11 -o firmware_vgg11.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running VGG-11 on QEMU RISC-V (with semihosting) ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_vgg11.elf -nographic -semihosting -semihosting-config enable=on,target=native

qemu-vgg11-compare: qemu-vgg11-packed qemu-vgg11-baseline
	@echo ""
	@echo "--- Completed QEMU comparison for packed and baseline VGG-11 ---"

qemu-vgg11-packed: $(COMMON_SRCS) swar_mac.h generated/vgg11_packed_weights.h \
        generated/vgg11_packed_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_inference.c swar_mac.c inference_ops.c generated/vgg11_packed_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_VGG11 -o firmware_vgg11_packed.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running VGG-11 Packed MQF on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_vgg11_packed.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

qemu-vgg11-baseline: $(COMMON_SRCS) swar_mac.h generated/vgg11_baseline_weights.h \
        generated/vgg11_baseline_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_inference.c swar_mac.c inference_ops.c generated/vgg11_baseline_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_VGG11 -DBASELINE_MODE -o firmware_vgg11_baseline.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running VGG-11 Baseline 8-bit on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_vgg11_baseline.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

qemu-resnet18-packed: swar_mac.h generated/resnet18_packed_weights.h \
        generated/resnet18_packed_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_resnet18_qemu.c swar_mac.c inference_ops.c generated/resnet18_packed_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_RESNET18 -o firmware_resnet18_packed.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running ResNet-18 Packed MQF on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_resnet18_packed.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

qemu-resnet18-baseline: swar_mac.h generated/resnet18_baseline_weights.h \
        generated/resnet18_baseline_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_resnet18_qemu.c swar_mac.c inference_ops.c generated/resnet18_baseline_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_RESNET18 -DBASELINE_MODE -o firmware_resnet18_baseline.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running ResNet-18 Baseline 8-bit on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_resnet18_baseline.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

all: resnet18 alexnet vgg11

# ── RISC-V cross-compile (ResNet-18 as default target) ───────────────────────
riscv: $(COMMON_SRCS) swar_mac.h generated/resnet18_packed_weights.h \
        generated/resnet18_packed_weights.cc
	$(CC_RISCV) $(CFLAGS_RISCV) -std=c99 $(COMMON_SRCS) \
	    generated/resnet18_packed_weights.cc \
	    -DTEST_RESNET18 -o firmware_resnet18.elf
	@echo "firmware_resnet18.elf built for RISC-V (rv32i)"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	del /Q test_resnet18.exe test_resnet18 \
	       test_alexnet.exe  test_alexnet  \
	       test_vgg11.exe    test_vgg11    \
	       test_vgg11_baseline.exe test_vgg11_baseline \
	       firmware_resnet18.elf firmware_vgg11.elf \
	       firmware_vgg11_packed.elf firmware_vgg11_baseline.elf *.o 2>nul || true

.PHONY: resnet18 alexnet vgg11 vgg11-baseline vgg11-compare qemu-vgg11 qemu-vgg11-packed qemu-vgg11-baseline qemu-vgg11-compare all riscv clean

qemu-alexnet-packed: swar_mac.h generated/alexnet_packed_weights.h generated/alexnet_packed_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_alexnet_qemu.c swar_mac.c inference_ops.c generated/alexnet_packed_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_ALEXNET -o firmware_alexnet_packed.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running AlexNet Packed MQF on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_alexnet_packed.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

qemu-alexnet-baseline: swar_mac.h generated/alexnet_baseline_weights.h generated/alexnet_baseline_weights.cc entry.s
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    entry.s main_alexnet_qemu.c swar_mac.c inference_ops.c generated/alexnet_baseline_weights.cc \
	    -T link.ld "-Wl,-e,_reset_start" -DTEST_ALEXNET -DBASELINE_MODE -o firmware_alexnet_baseline.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running AlexNet Baseline 8-bit on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_alexnet_baseline.elf -nographic -chardev stdio,id=stdio0 -semihosting-config enable=on,chardev=stdio0

swin_packed: generated/swin_packed_weights.h generated/swin_packed_weights.cc swin_transformer_packed.c
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. -I generated/ \
	    swin_transformer_packed.c generated/swin_packed_weights.cc \
	    -o firmware_swin_packed.elf --specs=semihost.specs
	@echo "firmware_swin_packed.elf built for RISC-V (rv32i)"

run_swin_packed: swin_packed
	@echo ""
	@echo "--- Running Swin Transformer (MHSA) Packed MQF on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_swin_packed.elf -nographic -semihosting -semihosting-config enable=on,target=native

qemu-swin-test: swar_mac.h
	$(CC_RISCV) -O2 -Wall -march=rv32im -mabi=ilp32 -std=c99 -I. \
	    main_swin_qemu.c swar_mac.c inference_ops.c \
	    -o firmware_swin_test.elf --specs=semihost.specs
	@echo ""
	@echo "--- Running Swin Transformer Isolated Kernels on QEMU RISC-V ---"
	qemu-system-riscv32 -machine virt -bios none -kernel firmware_swin_test.elf -nographic -semihosting -semihosting-config enable=on,target=native
