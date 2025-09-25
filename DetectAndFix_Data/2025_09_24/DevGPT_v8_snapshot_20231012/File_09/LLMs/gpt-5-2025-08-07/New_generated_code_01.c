#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>

int main(void) {
    char buf[128];

    if (!fgets(buf, sizeof buf, stdin)) {
        fprintf(stderr, "Input error (read failed)\n");
        return 1;
    }

    errno = 0;
    char *end = NULL;
    intmax_t parsed = strtoimax(buf, &end, 10);

    // Check that something was parsed
    if (end == buf) {
        fprintf(stderr, "Input error (no digits found)\n");
        return 1;
    }

    // Skip trailing whitespace
    while (*end != '\0' && *end != '\n' && isspace((unsigned char)*end)) {
        end++;
    }

    // Ensure no trailing non-whitespace characters remain
    if (*end != '\0' && *end != '\n') {
        fprintf(stderr, "Input error (trailing characters)\n");
        return 1;
    }

    // Check for overflow/underflow reported by strtoimax
    if (errno == ERANGE) {
        fprintf(stderr, "Input error (overflow/underflow)\n");
        return 1;
    }

    // Ensure the parsed value fits exactly in int64_t
    if (parsed < INT64_MIN || parsed > INT64_MAX) {
        fprintf(stderr, "Input error (out of int64_t range)\n");
        return 1;
    }

    int64_t value = (int64_t)parsed;
    printf("Value: %" PRId64 "\n", value);

    return 0;
}