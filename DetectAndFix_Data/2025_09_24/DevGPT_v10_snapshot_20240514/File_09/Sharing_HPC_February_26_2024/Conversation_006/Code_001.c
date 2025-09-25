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
        printf("Donde: <N> es el tamaño de la matriz y <imprimir> es 1 para imprimir la matriz resultante, 0 para no hacerlo.\n");
        return 1;
    }

    int N = atoi(argv[1]); // Convierte el primer argumento de la línea de comandos a entero
    int imprimir = atoi(argv[2]); // Convierte el segundo argumento a entero (1 para imprimir, 0 para no imprimir)

    // Asignación dinámica de matrices
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)calloc(N, sizeof(int)); // Inicializa a 0
    }

    srand(time(NULL)); // Inicializa el generador de números aleatorios

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

    // Imprime la matriz resultante C si imprimir es 1
    if (imprimir) {
        printf("Matriz resultante C (A*B):\n");
        printMatrix(C, N);
    }

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