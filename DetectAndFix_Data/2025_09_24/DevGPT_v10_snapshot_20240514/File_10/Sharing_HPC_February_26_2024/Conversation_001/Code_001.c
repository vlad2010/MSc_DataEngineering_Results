#include <stdio.h>
#include <stdlib.h>

#define N 4 // Define el tamaño de la matriz aquí

void printMatrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];

    // Inicializa las matrices A y B con algunos valores
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j; // Solo un ejemplo de inicialización
            B[i][j] = i - j; // Solo un ejemplo de inicialización
        }
    }

    // Inicializa la matriz C con ceros
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
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
    printMatrix(C);

    return 0;
}