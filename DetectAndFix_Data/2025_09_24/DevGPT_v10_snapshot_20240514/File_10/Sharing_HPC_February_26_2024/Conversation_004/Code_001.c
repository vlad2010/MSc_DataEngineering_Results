#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Para time()

#define N 4 // Define el tamaño de la matriz aquí

void printMatrix(int matrix[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    int i, j, k;

    // Inicializar el generador de números aleatorios
    srand(time(NULL)); // Usa el tiempo actual como semilla

    // Inicializa las matrices A y B con números aleatorios
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 100; // Números aleatorios entre 0 y 99
            B[i][j] = rand() % 100; // Números aleatorios entre 0 y 99
        }
    }

    // Inicializa la matriz C con ceros
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    // Multiplicación de matrices A y B, resultado en C
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Imprime la matriz resultante C
    printf("Matriz resultante C (A*B):\n");
    printMatrix(C);

    return 0;
}