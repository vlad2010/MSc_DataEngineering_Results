#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initializeAndFillMatrix(int ***matrix, int N, int isZero) {
    *matrix = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        (*matrix)[i] = (int *)malloc(N * sizeof(int));
        for (int j = 0; j < N; j++) {
            (*matrix)[i][j] = isZero ? 0 : rand() % 100;
        }
    }
}

void multiplyMatrices(int **A, int **B, int **C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0; // Initialize element
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void freeMatrix(int ***matrix, int N) {
    for (int i = 0; i < N; i++) {
        free((*matrix)[i]);
    }
    free(*matrix);
}

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
        printf("N value not provided\n");
        return -1;
    }

    int N = atoi(argv[1]);
    srand(time(NULL));

    int **A, **B, **C;

    initializeAndFillMatrix(&A, N, 0); // 0 for false, meaning fill with random numbers
    initializeAndFillMatrix(&B, N, 0);
    initializeAndFillMatrix(&C, N, 1); // 1 for true, meaning initialize to 0

    clock_t start = clock();
    multiplyMatrices(A, B, C, N);
    clock_t end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%d, %f\n", N, cpu_time_used);

    // Optionally print matrices if needed, can be controlled by a command-line flag
    // printMatrix(A, N);
    // printMatrix(B, N);
    // printMatrix(C, N);

    freeMatrix(&A, N);
    freeMatrix(&B, N);
    freeMatrix(&C, N);

    return 0;
}