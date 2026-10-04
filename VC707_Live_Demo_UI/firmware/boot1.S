.section .text.init
.globl _start
_start:
    la sp, _stack_top           # set up stack

    # Enable UART0 transmitter
    li t0, 0x64000008
    li t1, 1
    sw t1, 0(t0)
    
    #enable FPU
    csrr t0, mstatus      # read current mstatus
    li t1, 0x3000         # FS bits = 11b at bits 13:12
    or t0, t0, t1         # set FS=Dirty
    csrw mstatus, t0      # write back

    # Print boot message from assembly
    la a0, boot_msg

print_loop:
    lb t1, 0(a0)
    beqz t1, done
    li t0, 0x64000000          # UART0 TXDATA
wait_tx:
    lw t2, 0(t0)
    li t3, 0x80000000          # TX full mask
    and t2, t2, t3
    bne t2, zero, wait_tx
    sb t1, 0(t0)
    addi a0, a0, 1
    j print_loop
    
done:
    call main                  # jump to C main()
hang:
    j hang

.section .rodata
boot_msg:
    .asciz "Boot ROM running\n"

.section .bss
.align 8
stack_bottom:
    .space 0x100000       # 1 MB stack
_stack_top:
