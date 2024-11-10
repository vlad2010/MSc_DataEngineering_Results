#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

double dot_product(double* a, double* b, int size) {
    double result = 0.0;
    for(int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    double* a = (double*)malloc(SIZE * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Error allocating memory for vector a.\n");
        return EXIT_FAILURE;  // Exit program if allocation fails
    }

    double* b = (double*)malloc(SIZE * sizeof(double));
    if (b == NULL) {
        fprintf(stderr, "Error allocating memory for vector b.\n");
        free(a);  // Free previously allocated memory
        return EXIT_FAILURE;  // Exit program if allocation fails
    }

    // Initialize vectors
    for(int i = 0; i < SIZE; i++) {
        a[i] = (double)i;
        b[i] = (double)(SIZE - i);
    }

    double result = dot_product(a, b, SIZE);

    printf("Dot Product: %.2f\n", result);

    // Free allocated memory
    free(a);
    free(b);

    return 0;
}