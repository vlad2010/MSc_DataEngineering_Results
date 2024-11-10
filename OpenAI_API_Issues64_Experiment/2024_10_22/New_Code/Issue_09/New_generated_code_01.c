#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>

int main() {
    int64_t value;
    int64_t *p = &value;

    // Ensure input is successfully read into the variable 'value'
    if (scanf("%" SCNd64, p) != 1) {
        fprintf(stderr, "Input error\n");
        return EXIT_FAILURE; // Return a non-zero value indicating failure
    }

    printf("Value: %" PRId64 "\n", value);

    return EXIT_SUCCESS; // Return zero indicating success
}