#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>

#define MAX_PROCESSES 8 // Adjust based on your system's capability

void printMatrix(int *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

void multiplyChunk(int *A, int *B, int *C, int N, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0; // Initialize element
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 0, verbose = 0, numProcesses = MAX_PROCESSES;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1; // Enable verbose output
        } else if (i == 1) {
            N = atoi(argv[i]); // First positional argument is matrix size
        } else if (i == 2) {
            numProcesses = atoi(argv[i]); // Second positional argument is number of processes
        }
    }

    if (N == 0) {
        printf("Usage: %s <N> [num_processes] [-v|--verbose]\n", argv[0]);
        return 1;
    }

    // Adjust the number of processes if necessary
    numProcesses = (N < numProcesses) ? N : numProcesses;

    // Allocate shared memory for matrices A, B, C
    // Your shared memory allocation and initialization code goes here

    // Fork processes and perform matrix multiplication
    // Your process forking and multiplication code goes here

    // Wait for child processes to complete and collect results
    // Your process waiting and result collection code goes here

    if (verbose) {
        printf("Matrix A:\n");
        printMatrix(A, N);
        printf("\nMatrix B:\n");
        printMatrix(B, N);
        printf("\nResult Matrix C:\n");
        printMatrix(C, N);
    }

    // Output the computation time and cleanup shared memory
    // Your timing, output, and cleanup code goes here

    return 0;
}