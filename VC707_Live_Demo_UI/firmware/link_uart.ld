/* link.ld - separate sections to avoid RWX warnings */
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = 0x80000000, LENGTH = 256M
}

SECTIONS {
    /* Executable sections (text + rodata) */
    .text : {
        *(.text*)
        *(.text.init*)
        *(.rodata*)
    } > RAM

    /* Initialized data (RW) */
    .data : {
        *(.data*)
    } > RAM

    /* Zero-initialized data (RW) */
    .bss : {
        *(.bss*)
        *(COMMON)
    } > RAM
    
    /* Place binary image data in RAM */
    /*.image_bin : {*/
    /*KEEP(*(.rodata.image_bin*))*/
    /*} > RAM*/

    /* HTIF symbols */
    //.tohost   : { KEEP(*(.tohost)) } > RAM
    //.fromhost : { KEEP(*(.fromhost)) } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
    _stack_bottom = _stack_top - 0x100000;  /* 1 MB stack */
}
