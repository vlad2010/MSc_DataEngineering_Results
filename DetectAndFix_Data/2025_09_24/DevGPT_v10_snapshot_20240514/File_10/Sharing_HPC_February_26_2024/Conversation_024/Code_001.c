#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

#define N 4 // Tamaño de la matriz NxN

void multiplyRowByMatrix(int row[N], int matrix[N][N], int result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = 0;
        for (int j = 0; j < N; j++) {
            result[i] += row[j] * matrix[j][i];
        }
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N]; // Matrices
    pid_t pids[N]; // Array para almacenar los PIDs de los procesos hijos
    int status; // Estado de salida de los procesos hijos

    // Inicializar las matrices A y B con valores aleatorios
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    for (int i = 0; i < N; i++) {
        pids[i] = fork();
        if (pids[i] < 0) {
            perror("Fork falló");
            exit(EXIT_FAILURE);
        } else if (pids[i] == 0) { // Código del proceso hijo
            multiplyRowByMatrix(A[i], B, C[i]);
            exit(EXIT_SUCCESS); // El hijo termina después de realizar su tarea
        }
    }

    // Código del proceso padre, espera a que todos los hijos terminen
    for (int i = 0; i < N; i++) {
        waitpid(pids[i], &status, 0);
    }

    // Imprimir el resultado de la multiplicación de matrices
    printf("Matriz resultante C (A*B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}