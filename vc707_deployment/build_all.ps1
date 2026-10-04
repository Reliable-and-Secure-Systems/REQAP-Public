$CC = "C:\riscv-gcc\bin\riscv-none-elf-gcc.exe"
$CFLAGS = "-march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -I. -I../risc_v_backend -I../risc_v_backend/firmware"
$LDFLAGS = "-nostdlib -T link_uart.ld -lm -lgcc"

Write-Host "Cleaning old files..."
Remove-Item -Path *.elf, *.o -ErrorAction SilentlyContinue

Write-Host "Compiling boot and stubs..."
Invoke-Expression "$CC $CFLAGS -c boot1.S -o boot1.o"
Invoke-Expression "$CC $CFLAGS -c baremetal_libc_stubs.c -o baremetal_libc_stubs.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/firmware/inference_ops.c -o inference_ops.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/firmware/swar_mac.c -o swar_mac.o"

Write-Host "Compiling VGG11 Packed..."
Invoke-Expression "$CC $CFLAGS -DTEST_VGG11 -c main_vc707_vgg11.c -o main_vc707_vgg11.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/vgg11_packed_weights.cc -o vgg11_packed_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_vgg11.o vgg11_packed_weights.o inference_ops.o swar_mac.o -o vgg11_packed_vc707.elf"

Write-Host "Compiling VGG11 Baseline..."
Invoke-Expression "$CC $CFLAGS -DTEST_VGG11 -DBASELINE_MODE -c main_vc707_vgg11.c -o main_vc707_vgg11_base.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/vgg11_baseline_weights.cc -o vgg11_baseline_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_vgg11_base.o vgg11_baseline_weights.o inference_ops.o swar_mac.o -o vgg11_baseline_vc707.elf"

Write-Host "Compiling ResNet18 Packed..."
Invoke-Expression "$CC $CFLAGS -DTEST_RESNET18 -c main_vc707_resnet18.c -o main_vc707_resnet18.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/resnet18_packed_weights.cc -o resnet18_packed_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_resnet18.o resnet18_packed_weights.o inference_ops.o swar_mac.o -o resnet18_packed_vc707.elf"

Write-Host "Compiling ResNet18 Baseline..."
Invoke-Expression "$CC $CFLAGS -DTEST_RESNET18 -DBASELINE_MODE -c main_vc707_resnet18.c -o main_vc707_resnet18_base.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/resnet18_baseline_weights.cc -o resnet18_baseline_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_resnet18_base.o resnet18_baseline_weights.o inference_ops.o swar_mac.o -o resnet18_baseline_vc707.elf"

Write-Host "Compiling AlexNet Packed..."
Invoke-Expression "$CC $CFLAGS -DTEST_ALEXNET -c main_vc707_alexnet.c -o main_vc707_alexnet.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/alexnet_packed_weights.cc -o alexnet_packed_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_alexnet.o alexnet_packed_weights.o inference_ops.o swar_mac.o -o alexnet_packed_vc707.elf"

Write-Host "Compiling AlexNet Baseline..."
Invoke-Expression "$CC $CFLAGS -DTEST_ALEXNET -DBASELINE_MODE -c main_vc707_alexnet.c -o main_vc707_alexnet_base.o"
Invoke-Expression "$CC $CFLAGS -c ../risc_v_backend/generated/alexnet_baseline_weights.cc -o alexnet_baseline_weights.o"
Invoke-Expression "$CC $CFLAGS $LDFLAGS boot1.o baremetal_libc_stubs.o main_vc707_alexnet_base.o alexnet_baseline_weights.o inference_ops.o swar_mac.o -o alexnet_baseline_vc707.elf"

Write-Host "Done!"
