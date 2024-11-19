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
    double* b = (double*)malloc(SIZE * sizeof(double));

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
