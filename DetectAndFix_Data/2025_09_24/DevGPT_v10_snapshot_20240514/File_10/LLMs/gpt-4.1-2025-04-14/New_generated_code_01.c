#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

#define MAX_N 1000 // Limit to prevent resource exhaustion

void multiplyRowByMatrix(int **A, int **B, int **C, int N, int row) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[row][j] += A[row][k] * B[k][j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return -1;
    }

    // Use strtol for robust input parsing
    char *endptr;
    errno = 0;
    long n_long = strtol(argv[1], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || n_long <= 0 || n_long > MAX_N) {
        fprintf(stderr, "Invalid N. Please enter a positive integer <= %d.\n", MAX_N);
        return -1;
    }
    int N = (int)n_long;

    // Check for integer overflow in allocation
    if (N > 0 && N > INT_MAX / N / (int)sizeof(int)) {
        fprintf(stderr, "N is too large, would cause integer overflow.\n");
        return -1;
    }

    srand((unsigned int)time(NULL));

    // Shared memory allocation for matrices A, B, and C
    int shmidA = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);
    if (shmidA == -1) {
        perror("shmget A failed");
        return -1;
    }
    int shmidB = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);
    if (shmidB == -1) {
        perror("shmget B failed");
        shmctl(shmidA, IPC_RMID, NULL);
        return -1;
    }
    int shmidC = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);
    if (shmidC == -1) {
        perror("shmget C failed");
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        return -1;
    }

    int *shm_A = shmat(shmidA, NULL, 0);
    if (shm_A == (void *)-1) {
        perror("shmat A failed");
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        shmctl(shmidC, IPC_RMID, NULL);
        return -1;
    }
    int *shm_B = shmat(shmidB, NULL, 0);
    if (shm_B == (void *)-1) {
        perror("shmat B failed");
        shmdt(shm_A);
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        shmctl(shmidC, IPC_RMID, NULL);
        return -1;
    }
    int *shm_C = shmat(shmidC, NULL, 0);
    if (shm_C == (void *)-1) {
        perror("shmat C failed");
        shmdt(shm_A);
        shmdt(shm_B);
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        shmctl(shmidC, IPC_RMID, NULL);
        return -1;
    }

    // Convert flat arrays back to 2D arrays for easier handling
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        free(A); free(B); free(C);
        shmdt(shm_A); shmdt(shm_B); shmdt(shm_C);
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        shmctl(shmidC, IPC_RMID, NULL);
        return -1;
    }
    for (int i = 0; i < N; i++) {
        A[i] = shm_A + N*i;
        B[i] = shm_B + N*i;
        C[i] = shm_C + N*i;
    }

    // Filling matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
            C[i][j] = 0; // Initialize matrix C to 0
        }
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock(); // Start timing the multiplication

    // Fork a new process for each row in the result matrix C
    for (int i = 0; i < N; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            // Clean up and exit
            for (int j = 0; j < i; j++) wait(NULL); // Wait for already forked children
            free(A); free(B); free(C);
            shmdt(shm_A); shmdt(shm_B); shmdt(shm_C);
            shmctl(shmidA, IPC_RMID, NULL);
            shmctl(shmidB, IPC_RMID, NULL);
            shmctl(shmidC, IPC_RMID, NULL);
            return -1;
        }
        if (pid == 0) { // Child process
            multiplyRowByMatrix(A, B, C, N, i);
            free(A); free(B); free(C);
            shmdt(shm_A); shmdt(shm_B); shmdt(shm_C);
            exit(0); // Child process exits after computing its row
        }
    }

    // Wait for all child processes to complete
    while (wait(NULL) > 0);

    end = clock(); // End timing the multiplication
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Matrix multiplication completed in %f seconds\n", cpu_time_used);

    // Detach and remove shared memory
    shmdt(shm_A);
    shmdt(shm_B);
    shmdt(shm_C);
    shmctl(shmidA, IPC_RMID, NULL);
    shmctl(shmidB, IPC_RMID, NULL);
    shmctl(shmidC, IPC_RMID, NULL);

    free(A);
    free(B);
    free(C);

    return 0;
}