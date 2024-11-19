#include <omp.h>

double dot_product(double* a, double* b, int size) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}
