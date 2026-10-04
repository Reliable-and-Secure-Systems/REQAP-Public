#include <stdio.h>
#include <stdlib.h>

extern char __bss_start;
extern char __BSS_END__;

int main(void);

void entry_point(void) {
    // Clear BSS
    char *bss = &__bss_start;
    while (bss < &__BSS_END__) {
        *bss++ = 0;
    }
    
    main();
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("HELLO_TEST_START\n");
    printf("Hello RISC-V Semihosting with custom startup!\n");
    printf("HELLO_TEST_END\n");
    exit(0);
    return 0;
}
