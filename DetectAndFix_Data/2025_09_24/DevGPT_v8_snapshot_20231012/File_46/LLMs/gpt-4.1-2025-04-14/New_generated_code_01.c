#include <stdio.h>
#include <stdlib.h>
#include <limits.h> // For SIZE_MAX

#define SIZE 1000000

double dot_product(double* a, double* b, int size) {
    double result = 0.0;
    for(int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    // Check for integer overflow in allocation size
    if (SIZE > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Requested allocation size is too large.\n");
        return 1;
    }

    double* a = (double*)malloc(SIZE * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Memory allocation failed for vector a.\n");
        return 1;
    }

    double* b = (double*)malloc(SIZE * sizeof(double));
    if (b == NULL) {
        fprintf(stderr, "Memory allocation failed for vector b.\n");
        free(a); // Free previously allocated memory
        return 1;
    }

    // Initialize vectors
    for(int i = 0; i < SIZE; i++) {
        a[i] = (double)i;
        b[i] = (double)(SIZE - i);
    }

    double result = dot_product(a, b, SIZE);

    printf("Dot Product: %.2f\n", result);

    free(a);
    free(b);

    return 0;
}