#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#define MAX_PROCESSORS 4 // Example limit, adjust based on your system

// Helper function to safely parse positive integers
int parse_positive_int(const char *str, int *result) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if (endptr == str || *endptr != '\0' || val <= 0 || val > INT_MAX) {
        return 0;
    }
    *result = (int)val;
    return 1;
}

void allocateSharedMatrix(int *shmid, int N) {
    // Check for integer overflow
    if (N <= 0 || N > 46340) { // sqrt(INT_MAX/sizeof(int)) ~ 46340 for 32-bit int
        fprintf(stderr, "Invalid matrix size for shared memory allocation.\n");
        exit(1);
    }
    size_t size = (size_t)N * (size_t)N * sizeof(int);
    if (size / N / sizeof(int) != (size_t)N) {
        fprintf(stderr, "Integer overflow detected in shared memory allocation.\n");
        exit(1);
    }
    *shmid = shmget(IPC_PRIVATE, size, IPC_CREAT | 0600); // Use 0600 for least privilege
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
    if (!matrix) {
        perror("malloc");
        shmdt(data);
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        matrix[i] = data + N * i;
    }
    return matrix;
}

void detachMatrix(int **matrix) {
    if (matrix && *matrix) {
        if (shmdt(*matrix) == -1) {
            perror("shmdt");
        }
    }
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

    int N;
    if (!parse_positive_int(argv[1], &N)) {
        fprintf(stderr, "Invalid matrix size N: %s\n", argv[1]);
        exit(1);
    }

    int numProcessors = MAX_PROCESSORS; // Default value

    // Optional: parse -p num_processors
    if (argc >= 4 && strcmp(argv[2], "-p") == 0) {
        if (!parse_positive_int(argv[3], &numProcessors) || numProcessors < 1 || numProcessors > MAX_PROCESSORS) {
            fprintf(stderr, "Invalid number of processors: %s\n", argv[3]);
            exit(1);
        }
    }
    if (numProcessors > N) {
        numProcessors = N; // Don't use more processors than rows
    }

    srand((unsigned int)time(NULL));

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

    int rowsPerProcessor = N / numProcessors;

    for (int p = 0; p < numProcessors; p++) {
        int startRow = p * rowsPerProcessor;
        int endRow = (p + 1) * rowsPerProcessor;
        if (p == numProcessors - 1) {
            endRow = N; // Last processor gets the remaining rows
        }

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            // Clean up shared memory before exit
            detachMatrix(A);
            detachMatrix(B);
            detachMatrix(C);
            shmctl(shmidA, IPC_RMID, NULL);
            shmctl(shmidB, IPC_RMID, NULL);
            shmctl(shmidC, IPC_RMID, NULL);
            exit(1);
        }
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