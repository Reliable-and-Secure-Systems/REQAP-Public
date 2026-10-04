#include <stdio.h>
#include <stdint.h>
int main(void) {
    printf("SEMHOST_TEST_START\n");
    FILE *f = fopen("semihost_test_output.csv", "w");
    if (f) {
        fprintf(f, "METRIC_CYCLES=123\n");
        fprintf(f, "METRIC_INSTRUCTIONS=456\n");
        fprintf(f, "METRIC_MEMORY_LOADS=789\n");
        fclose(f);
        printf("FILE_WRITTEN\n");
    } else {
        printf("FILE_OPEN_FAILED\n");
    }
    printf("SEMHOST_TEST_END\n");
    return 0;
}
