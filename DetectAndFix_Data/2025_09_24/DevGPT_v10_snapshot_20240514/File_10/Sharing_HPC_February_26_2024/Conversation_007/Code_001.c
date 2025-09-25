#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <N> <imprimir>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int imprimir = atoi(argv[2]);

    // Asignación dinámica de matrices
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)calloc(N, sizeof(int));
    }

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock(); // Inicia el cronómetro

    // Multiplicación de matrices A y B, resultado en C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = clock(); // Detiene el cronómetro

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Calcula el tiempo transcurrido

    // Imprime la matriz resultante C si imprimir es 1
    if (imprimir) {
        printf("Matriz resultante C (A*B):\n");
        printMatrix(C, N);
    }

    printf("Tiempo de CPU utilizado: %f segundos\n", cpu_time_used);

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