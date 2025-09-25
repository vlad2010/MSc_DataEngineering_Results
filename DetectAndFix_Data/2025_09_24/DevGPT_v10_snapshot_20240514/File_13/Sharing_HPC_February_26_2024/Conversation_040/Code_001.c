#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <string.h>

#define MAX_PROCESSORS 4 // Example limit, adjust based on your system

void allocateSharedMatrix(int *shmid, int N) {
    *shmid = shmget(IPC_PRIVATE, N * N * sizeof(int), IPC_CREAT | 0666);
    if (*shmid < 0) {
        perror("shmget");
        exit(1);
    }
}

int** attachMatrix(int shmid, int N) {
    int *data = (int *)shmat(shmid, NULL, 0);
    if (data == (int *)-1) {
        perror("shmat");
        exit(1);
    }

    int **matrix = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        matrix[i] = data + N * i;
    }
    return matrix;
}

void detachMatrix(int **matrix) {
    shmdt(*matrix);
    free(matrix);
}

void fillMatrix(int **matrix, int N, int isRandom) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = isRandom ? rand() % 100 : 0;
        }
    }
}

void multiplyMatrixChunk(int **A, int **B, int **C, int N, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> [-p num_processors]\n", argv[0]);
        exit(1);
    }

    int N = atoi(argv[1]);
    srand(time(NULL));

    int shmidA, shmidB, shmidC;
    allocateSharedMatrix(&shmidA, N);
    allocateSharedMatrix(&shmidB, N);
    allocateSharedMatrix(&shmidC, N);

    int **A = attachMatrix(shmidA, N);
    int **B = attachMatrix(shmidB, N);
    int **C = attachMatrix(shmidC, N);

    fillMatrix(A, N, 1); // Fill A and B with random numbers
    fillMatrix(B, N, 1);
    fillMatrix(C, N, 0); // Initialize C to zeros

    int numProcessors = MAX_PROCESSORS; // Default value
    int rowsPerProcessor = N / numProcessors;

    for (int p = 0; p < numProcessors; p++) {
        int startRow = p * rowsPerProcessor;
        int endRow = (p + 1) * rowsPerProcessor;
        if (p == numProcessors - 1) {
            endRow = N; // Last processor gets the remaining rows
        }

        pid_t pid = fork();
        if (pid == 0) { // Child process
            multiplyMatrixChunk(A, B, C, N, startRow, endRow);
            detachMatrix(A);
            detachMatrix(B);
            detachMatrix(C);
            exit(0);
        }
    }

    while (wait(NULL) > 0); // Wait for all child processes to finish

    // Optionally print the result matrix C here if needed

    // Cleanup
    detachMatrix(A);
    detachMatrix(B);
    detachMatrix(C);
    shmctl(shmidA, IPC_RMID, NULL);
    shmctl(shmidB, IPC_RMID, NULL);
    shmctl(shmidC, IPC_RMID, NULL);

    return 0;
}