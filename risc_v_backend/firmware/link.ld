OUTPUT_ARCH( "riscv" )
ENTRY( _start )

SECTIONS
{
  . = 0x80000000;
  
  .text : {
    KEEP(*(.init))
    *(.text .text.*)
  }
  
  .rodata : { 
    *(.rodata .rodata.*) 
  }
  
  .data : { 
    *(.data .data.*) 
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.sdata .sdata.*) 
  }
  
  .bss : {
    __bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    __BSS_END__ = .;
  }
  
  _end = .;
  PROVIDE(__stack = 0x84000000);
  PROVIDE(_stack = 0x84000000);
}
