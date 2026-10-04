.global _reset_start
.section .init
_reset_start:
    # Set stack pointer (sp) to a valid location in RAM (64MB into 128MB RAM)
    lui sp, 0x84000
    
    # Jump to standard C library runtime startup
    j _start
