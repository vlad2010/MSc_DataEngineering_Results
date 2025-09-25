#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

#define MAX_N 2048  /* Upper bound to avoid uncontrolled resource consumption (CWE-400/CWE-789) */
#define IDX(i, j, n) ((i) * (n) + (j))

/* Safely parse a positive size_t from string */
static int parse_n(const char *s, size_t *out) {
    if (!s || !out) return -1;

    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') {
        return -1; /* invalid or out of range */
    }
    if (v == 0 || v > SIZE_MAX) {
        return -1; /* disallow zero and values beyond size_t */
    }
    *out = (size_t)v;
    return 0;
}

/* Check for overflow in n*n elements of type int64_t */
static int check_square_allocation_overflow(size_t n) {
    if (n == 0) return -1;
    size_t limit_elems = SIZE_MAX / sizeof(int64_t);
    if (n > limit_elems / n) {
        return -1; /* n * n would overflow */
    }
    return 0;
}

static void printMatrix(const int64_t *matrix, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%" PRId64 "\t", matrix[IDX(i, j, N)]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <N>\n", argv[0]);
        return 1;
    }

    size_t N = 0;
    if (parse_n(argv[1], &N) != 0) {
        fprintf(stderr, "Error: N debe ser un entero positivo válido dentro de rango.\n");
        return 1;
    }

    if (N > MAX_N) {
        fprintf(stderr, "Error: N=%zu excede el máximo permitido (%d) para evitar consumo excesivo de recursos.\n",
                N, MAX_N);
        return 1;
    }

    if (check_square_allocation_overflow(N) != 0) {
        fprintf(stderr, "Error: N es demasiado grande y provocaría desbordamiento de tamaño de asignación.\n");
        return 1;
    }

    size_t elems = N * N;

    int64_t *A = (int64_t *)malloc(elems * sizeof(int64_t));
    int64_t *B = (int64_t *)malloc(elems * sizeof(int64_t));
    int64_t *C = (int64_t *)calloc(elems, sizeof(int64_t)); /* Initialize to zero */

    if (!A || !B || !C) {
        fprintf(stderr, "Error: fallo al asignar memoria.\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    /* Initialize PRNG (not cryptographically secure; acceptable for this demo) */
    srand((unsigned int)time(NULL));

    /* Fill A and B with pseudo-random numbers in [0, 99] */
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[IDX(i, j, N)] = (int64_t)(rand() % 100);
            B[IDX(i, j, N)] = (int64_t)(rand() % 100);
        }
    }

    /* C = A * B (matrix multiplication) using 64-bit arithmetic to avoid overflow */
    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < N; k++) {
            int64_t a_ik = A[IDX(i, k, N)];
            for (size_t j = 0; j < N; j++) {
                C[IDX(i, j, N)] += a_ik * B[IDX(k, j, N)];
            }
        }
    }

    printf("Matriz resultante C (A*B):\n");
    printMatrix(C, N);

    free(A);
    free(B);
    free(C);
    return 0;
}