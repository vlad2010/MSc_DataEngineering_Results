#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#define MAX_PROCESSORS 4
#define MAX_MATRIX_SIZE 1000
#define MIN_MATRIX_SIZE 1

// Semaphore operations
void sem_wait(int semid, int semnum) {
    struct sembuf sb = {semnum, -1, 0};
    if (semop(semid, &sb, 1) == -1) {
        perror("semop wait");
        exit(1);
    }
}

void sem_signal(int semid, int semnum) {
    struct sembuf sb = {semnum, 1, 0};
    if (semop(semid, &sb, 1) == -1) {
        perror("semop signal");
        exit(1);
    }
}

int allocateSharedMatrix(int N) {
    // Check for integer overflow
    size_t total_size = (size_t)N * (size_t)N * sizeof(int);
    if (N > 0 && total_size / N / sizeof(int) != (size_t)N) {
        fprintf(stderr, "Matrix size would cause integer overflow\n");
        exit(1);
    }
    
    int shmid = shmget(IPC_PRIVATE, total_size, IPC_CREAT | 0600); // More restrictive permissions
    if (shmid < 0) {
        perror("shmget");
        exit(1);
    }
    return shmid;
}

int** attachMatrix(int shmid, int N) {
    int *data = (int *)shmat(shmid, NULL, 0);
    if (data == (int *)-1) {
        perror("shmat");
        exit(1);
    }

    int **matrix = (int **)calloc(N, sizeof(int *));
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
        shmdt(*matrix);
        free(matrix);
    }
}

void fillMatrix(int **matrix, int N, int isRandom) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = isRandom ? rand() % 100 : 0;
        }
    }
}

void multiplyMatrixChunk(int **A, int **B, int **C, int N, int startRow, int endRow, int semid) {
    // Local computation buffer to avoid race conditions
    int **localC = (int **)calloc(endRow - startRow, sizeof(int *));
    if (!localC) {
        perror("malloc localC");
        exit(1);
    }
    
    for (int i = 0; i < endRow - startRow; i++) {
        localC[i] = (int *)calloc(N, sizeof(int));
        if (!localC[i]) {
            perror("malloc localC row");
            exit(1);
        }
    }
    
    // Compute locally
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            long long sum = 0; // Use long long to prevent overflow
            for (int k = 0; k < N; k++) {
                sum += (long long)A[i][k] * B[k][j];
            }
            // Check for integer overflow
            if (sum > INT_MAX || sum < INT_MIN) {
                fprintf(stderr, "Integer overflow in matrix multiplication\n");
                exit(1);
            }
            localC[i - startRow][j] = (int)sum;
        }
    }
    
    // Write to shared memory with synchronization
    sem_wait(semid, 0);
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = localC[i - startRow][j];
        }
    }
    sem_signal(semid, 0);
    
    // Cleanup local buffer
    for (int i = 0; i < endRow - startRow; i++) {
        free(localC[i]);
    }
    free(localC);
}

int parseAndValidateInt(const char *str, const char *name) {
    char *endptr;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    
    if (errno != 0 || *endptr != '\0' || str == endptr) {
        fprintf(stderr, "Invalid %s: not a valid integer\n", name);
        exit(1);
    }
    
    if (val > INT_MAX || val < INT_MIN) {
        fprintf(stderr, "Invalid %s: integer overflow\n", name);
        exit(1);
    }
    
    return (int)val;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> [-p num_processors]\n", argv[0]);
        exit(1);
    }

    int N = parseAndValidateInt(argv[1], "matrix size");
    
    if (N < MIN_MATRIX_SIZE || N > MAX_MATRIX_SIZE) {
        fprintf(stderr, "Matrix size must be between %d and %d\n", 
                MIN_MATRIX_SIZE, MAX_MATRIX_SIZE);
        exit(1);
    }

    int numProcessors = MAX_PROCESSORS;
    
    // Parse optional processor count
    if (argc >= 4 && strcmp(argv[2], "-p") == 0) {
        numProcessors = parseAndValidateInt(argv[3], "processor count");
        if (numProcessors < 1 || numProcessors > MAX_PROCESSORS) {
            fprintf(stderr, "Number of processors must be between 1 and %d\n", MAX_PROCESSORS);
            exit(1);
        }
    }
    
    // Ensure we don't have more processors than rows
    if (numProcessors > N) {
        numProcessors = N;
    }

    srand(time(NULL));

    // Create semaphore for synchronization
    int semid = semget(IPC_PRIVATE, 1, IPC_CREAT | 0600);
    if (semid < 0) {
        perror("semget");
        exit(1);
    }
    
    // Initialize semaphore
    if (semctl(semid, 0, SETVAL, 1) == -1) {
        perror("semctl");
        exit(1);
    }

    int shmidA = allocateSharedMatrix(N);
    int shmidB = allocateSharedMatrix(N);
    int shmidC = allocateSharedMatrix(N);

    int **A = attachMatrix(shmidA, N);
    int **B = attachMatrix(shmidB, N);
    int **C = attachMatrix(shmidC, N);

    fillMatrix(A, N, 1);
    fillMatrix(B, N, 1);
    fillMatrix(C, N, 0);

    int rowsPerProcessor = N / numProcessors;
    pid_t *pids = (pid_t *)calloc(numProcessors, sizeof(pid_t));
    if (!pids) {
        perror("malloc pids");
        exit(1);
    }

    for (int p = 0; p < numProcessors; p++) {
        int startRow = p * rowsPerProcessor;
        int endRow = (p + 1) * rowsPerProcessor;
        if (p == numProcessors - 1) {
            endRow = N;
        }

        pids[p] = fork();
        if (pids[p] < 0) {
            perror("fork");
            // Kill already created children
            for (int i = 0; i < p; i++) {
                kill(pids[i], SIGTERM);
            }
            free(pids);
            exit(1);
        }
        
        if (pids[p] == 0) { // Child process
            multiplyMatrixChunk(A, B, C, N, startRow, endRow, semid);
            detachMatrix(A);
            detachMatrix(B);
            detachMatrix(C);
            exit(0);
        }
    }

    // Wait for all child processes
    int status;
    for (int i = 0; i < numProcessors; i++) {
        if (waitpid(pids[i], &status, 0) == -1) {
            perror("waitpid");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed\n", i);
        }
    }
    
    free(pids);

    // Cleanup
    detachMatrix(A);
    detachMatrix(B);
    detachMatrix(C);
    
    if (shmctl(shmidA, IPC_RMID, NULL) == -1) perror("shmctl A");
    if (shmctl(shmidB, IPC_RMID, NULL) == -1) perror("shmctl B");
    if (shmctl(shmidC, IPC_RMID, NULL) == -1) perror("shmctl C");
    if (semctl(semid, 0, IPC_RMID) == -1) perror("semctl");

    return 0;
}