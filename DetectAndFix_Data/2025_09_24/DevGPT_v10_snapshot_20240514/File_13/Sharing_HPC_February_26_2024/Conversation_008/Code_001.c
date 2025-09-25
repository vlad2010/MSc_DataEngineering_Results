#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);

    // Asignación dinámica de matrices
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)calloc(N, sizeof(int)); // Inicializa a 0
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

    // Formato CSV: N, tiempo_de_CPU
    printf("%d, %f\n", N, cpu_time_used);

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