    #include <stdio.h>
    #include <stdlib.h>

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

        // Inicializa las matrices A y B con algunos valores
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = i + j; // Solo un ejemplo de inicialización
                B[i][j] = i - j; // Solo un ejemplo de inicialización
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