#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <limits.h>

#define MAX_MATRIX_SIZE 1000 // Set a reasonable upper limit

void printMatrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <N>\n", argv[0]);
        return 1;
    }

    // Robust input parsing
    char *endptr;
    errno = 0;
    long n_long = strtol(argv[1], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || n_long <= 0 || n_long > MAX_MATRIX_SIZE) {
        fprintf(stderr, "Error: N debe ser un entero positivo entre 1 y %d.\n", MAX_MATRIX_SIZE);
        return 1;
    }
    int N = (int)n_long;

    // Dynamic allocation with error checking
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    if (!A || !B || !C) {
        fprintf(stderr, "Error: No se pudo asignar memoria para las matrices principales.\n");
        free(A); free(B); free(C);
        return 1;
    }

    int allocation_failed = 0;
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)calloc(N, sizeof(int)); // Inicializa a 0
        if (!A[i] || !B[i] || !C[i]) {
            allocation_failed = 1;
            // Free previously allocated memory
            for (int j = 0; j <= i; j++) {
                free(A[j]);
                free(B[j]);
                free(C[j]);
            }
            free(A); free(B); free(C);
            fprintf(stderr, "Error: No se pudo asignar memoria para las filas de las matrices.\n");
            return 1;
        }
    }

    srand((unsigned int)time(NULL)); // Inicializa el generador de números aleatorios

    // Llena las matrices A y B con números aleatorios
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    // Multiplicación de matrices A y B, resultado en C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Imprime la matriz resultante C
    printf("Matriz resultante C (A*B):\n");
    printMatrix(C, N);

    // Liberar memoria
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}