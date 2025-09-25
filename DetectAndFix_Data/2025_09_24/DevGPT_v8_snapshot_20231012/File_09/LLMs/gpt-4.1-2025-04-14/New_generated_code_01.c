#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>

#define INPUT_BUFFER_SIZE 100

int main() {
    char input[INPUT_BUFFER_SIZE];
    int64_t value;
    char *endptr;

    printf("Enter a 64-bit integer: ");
    if (fgets(input, sizeof(input), stdin) == NULL) {
        printf("Input error\n");
        return 1;
    }

    // Remove trailing newline if present
    input[strcspn(input, "\n")] = 0;

    errno = 0;
    value = strtoll(input, &endptr, 10);

    // Check for conversion errors
    if (errno == ERANGE || value < INT64_MIN || value > INT64_MAX) {
        printf("Value out of range for int64_t\n");
        return 1;
    }
    if (endptr == input || *endptr != '\0') {
        printf("Invalid input: not a valid 64-bit integer\n");
        return 1;
    }

    printf("Value: %" PRId64 "\n", value);

    return 0;
}