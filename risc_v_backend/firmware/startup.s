.global _start
.section .init
_start:
    # Set stack pointer (sp) to a valid location in RAM.
    # QEMU virt RAM starts at 0x80000000. 0x84000000 is 64MB into RAM.
    lui sp, 0x84000
    
    # Initialize gp (global pointer)
    .option push
    .option norelax
    lui gp, %hi(__global_pointer$)
    addi gp, gp, %lo(__global_pointer$)
    .option pop

    # Call wrapper function to clear BSS and execute main
    jal entry_point

    # In case entry_point returns, call exit(0)
    li a0, 0
    jal exit

    # Loop forever if exit does not terminate
1:  j 1b
