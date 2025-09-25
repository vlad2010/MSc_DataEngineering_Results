#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>

#define SIZE ((size_t)1000000u)

static int dot_product(const double *a, const double *b, size_t size, double *out_result) {
    if (!out_result) {
        return EINVAL;
    }
    if (size > 0 && (!a || !b)) {
        return EINVAL; // prevent NULL dereference
    }

    double acc = 0.0;
    for (size_t i = 0; i < size; i++) {
        acc += a[i] * b[i];
    }
    *out_result = acc;
    return 0;
}

int main(void) {
    // Check for overflow in allocation size calculation
    if (SIZE > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Requested SIZE causes overflow in allocation size\n");
        return EXIT_FAILURE;
    }

    double *a = malloc(SIZE * sizeof(*a));
    double *b = malloc(SIZE * sizeof(*b));

    if (!a || !b) {
        fprintf(stderr, "Memory allocation failed\n");
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    // Initialize vectors
    for (size_t i = 0; i < SIZE; i++) {
        a[i] = (double)i;
        b[i] = (double)(SIZE - i);
    }

    double result = 0.0;
    int rc = dot_product(a, b, SIZE, &result);
    if (rc != 0) {
        fprintf(stderr, "dot_product failed: %d\n", rc);
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    printf("Dot Product: %.2f\n", result);

    free(a);
    free(b);
    return EXIT_SUCCESS;
}