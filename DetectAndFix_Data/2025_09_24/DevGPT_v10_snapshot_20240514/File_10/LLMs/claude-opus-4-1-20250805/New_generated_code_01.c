#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/sem.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <string.h>

#define MAX_MATRIX_SIZE 1000
#define MIN_MATRIX_SIZE 1
#define MAX_PROCESSES 100

// Semaphore operations for synchronization
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

void multiplyRowByMatrix(int **A, int **B, int **C, int N, int row, int semid) {
    int *temp_row = malloc(N * sizeof(int));
    if (!temp_row) {
        perror("malloc temp_row");
        exit(1);
    }
    
    // Calculate in local memory first
    for (int j = 0; j < N; j++) {
        temp_row[j] = 0;
        for (int k = 0; k < N; k++) {
            // Check for integer overflow in multiplication
            if (A[row][k] != 0 && B[k][j] != 0) {
                if (abs(A[row][k]) > INT_MAX / abs(B[k][j])) {
                    fprintf(stderr, "Integer overflow detected in multiplication\n");
                    free(temp_row);
                    exit(1);
                }
            }
            int product = A[row][k] * B[k][j];
            
            // Check for integer overflow in addition
            if ((temp_row[j] > 0 && product > INT_MAX - temp_row[j]) ||
                (temp_row[j] < 0 && product < INT_MIN - temp_row[j])) {
                fprintf(stderr, "Integer overflow detected in addition\n");
                free(temp_row);
                exit(1);
            }
            temp_row[j] += product;
        }
    }
    
    // Use semaphore for synchronized write
    sem_wait(semid, 0);
    for (int j = 0; j < N; j++) {
        C[row][j] = temp_row[j];
    }
    sem_signal(semid, 0);
    
    free(temp_row);
}

int safe_atoi(const char *str, int *result) {
    char *endptr;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    
    if (errno != 0 || endptr == str || *endptr != '\0') {
        return -1;
    }
    
    if (val < INT_MIN || val > INT_MAX) {
        return -1;
    }
    
    *result = (int)val;
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N>\n", argv[0]);
        printf("Where N is matrix size (1-%d)\n", MAX_MATRIX_SIZE);
        return -1;
    }

    int N;
    if (safe_atoi(argv[1], &N) != 0) {
        fprintf(stderr, "Error: Invalid input for matrix size\n");
        return -1;
    }
    
    if (N < MIN_MATRIX_SIZE || N > MAX_MATRIX_SIZE) {
        fprintf(stderr, "Error: Matrix size must be between %d and %d\n", 
                MIN_MATRIX_SIZE, MAX_MATRIX_SIZE);
        return -1;
    }

    // Check for overflow in size calculation
    size_t total_size;
    if (__builtin_umull_overflow(N, N, &total_size) ||
        __builtin_umull_overflow(total_size, sizeof(int), &total_size)) {
        fprintf(stderr, "Error: Matrix size too large (overflow)\n");
        return -1;
    }

    srand(time(NULL));

    // Create semaphore for synchronization
    int semid = semget(IPC_PRIVATE, 1, IPC_CREAT | 0666);
    if (semid == -1) {
        perror("semget");
        return -1;
    }
    
    // Initialize semaphore
    if (semctl(semid, 0, SETVAL, 1) == -1) {
        perror("semctl");
        semctl(semid, 0, IPC_RMID);
        return -1;
    }

    // Shared memory allocation with error checking
    int shmidA = shmget(IPC_PRIVATE, total_size, IPC_CREAT | 0666);
    if (shmidA == -1) {
        perror("shmget A");
        semctl(semid, 0, IPC_RMID);
        return -1;
    }
    
    int shmidB = shmget(IPC_PRIVATE, total_size, IPC_CREAT | 0666);
    if (shmidB == -1) {
        perror("shmget B");
        shmctl(shmidA, IPC_RMID, NULL);
        semctl(semid, 0, IPC_RMID);
        return -1;
    }
    
    int shmidC = shmget(IPC_PRIVATE, total_size, IPC_CREAT | 0666);
    if (shmidC == -1) {
        perror("shmget C");
        shmctl(shmidA, IPC_RMID, NULL);
        shmctl(shmidB, IPC_RMID, NULL);
        semctl(semid, 0, IPC_RMID);
        return -1;
    }

    int *shm_A = shmat(shmidA, NULL, 0);
    if (shm_A == (void *)-1) {
        perror("shmat A");
        goto cleanup_shm;
    }
    
    int *shm_B = shmat(shmidB, NULL, 0);
    if (shm_B == (void *)-1) {
        perror("shmat B");
        shmdt(shm_A);
        goto cleanup_shm;
    }
    
    int *shm_C = shmat(shmidC, NULL, 0);
    if (shm_C == (void *)-1) {
        perror("shmat C");
        shmdt(shm_A);
        shmdt(shm_B);
        goto cleanup_shm;
    }

    // Allocate 2D array pointers with error checking
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    
    if (!A || !B || !C) {
        perror("malloc arrays");
        free(A);
        free(B);
        free(C);
        shmdt(shm_A);
        shmdt(shm_B);
        shmdt(shm_C);
        goto cleanup_shm;
    }
    
    for (int i = 0; i < N; i++) {
        A[i] = shm_A + N*i;
        B[i] = shm_B + N*i;
        C[i] = shm_C + N*i;
    }

    // Initialize matrices with bounded random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
            C[i][j] = 0;
        }
    }

    clock_t start = clock();
    
    // Limit number of concurrent processes
    int max_concurrent = (N < MAX_PROCESSES) ? N : MAX_PROCESSES;
    int current_row = 0;
    int active_processes = 0;
    
    while (current_row < N || active_processes > 0) {
        // Fork new processes up to the limit
        while (current_row < N && active_processes < max_concurrent) {
            pid_t pid = fork();
            if (pid == -1) {
                perror("fork");
                // Wait for existing processes before continuing
                while (wait(NULL) > 0) active_processes--;
                continue;
            } else if (pid == 0) {
                // Child process
                multiplyRowByMatrix(A, B, C, N, current_row, semid);
                exit(0);
            } else {
                // Parent process
                active_processes++;
                current_row++;
            }
        }
        
        // Wait for at least one child to finish
        if (active_processes >= max_concurrent || current_row >= N) {
            if (wait(NULL) > 0) {
                active_processes--;
            }
        }
    }

    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Matrix multiplication completed in %f seconds\n", cpu_time_used);

    // Cleanup
    free(A);
    free(B);
    free(C);
    
    shmdt(shm_A);
    shmdt(shm_B);
    shmdt(shm_C);

cleanup_shm:
    shmctl(shmidA, IPC_RMID, NULL);
    shmctl(shmidB, IPC_RMID, NULL);
    shmctl(shmidC, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);

    return 0;
}