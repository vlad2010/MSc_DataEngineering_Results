#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>

void multiplyRowByMatrix(int *A, int *B, int *C, int N, int row) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[row * N + j] += A[row * N + k] * B[k * N + j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]); // Converts the command line argument to an integer
    srand(time(NULL));

    // Shared memory allocation for matrices A, B, and C
    int shmidA = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);
    int shmidB = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);
    int shmidC = shmget(IPC_PRIVATE, N*N*sizeof(int), IPC_CREAT | 0666);

    int *A = shmat(shmidA, NULL, 0);
    int *B = shmat(shmidB, NULL, 0);
    int *C = shmat(shmidC, NULL, 0);

    // Filling matrices A and B with random values
    for (int i = 0; i < N*N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
        C[i] = 0; // Initialize matrix C to 0
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock(); // Start timing the multiplication

    // Fork a new process for each row in the result matrix C
    for (int i = 0; i < N; i++) {
        pid_t pid = fork();
        if (pid == 0) { // Child process
            multiplyRowByMatrix(A, B, C, N, i);
            shmdt(A);
            shmdt(B);
            shmdt(C);
            exit(0); // Child process exits after computing its row
        }
    }

    // Wait for all child processes to complete
    while (wait(NULL) > 0);

    end = clock(); // End timing the multiplication
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Output in CSV format
    printf("%d, %f\n", N, cpu_time_used);

    // Detach and remove shared memory
    shmdt(A);
    shmdt(B);
    shmdt(C);
    shmctl(shmidA, IPC_RMID, NULL);
    shmctl(shmidB, IPC_RMID, NULL);
    shmctl(shmidC, IPC_RMID, NULL);

    return 0;
}