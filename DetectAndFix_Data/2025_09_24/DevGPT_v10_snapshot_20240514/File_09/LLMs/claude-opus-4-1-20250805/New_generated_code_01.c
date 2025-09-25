#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <errno.h>
#include <string.h>

#define MAX_MATRIX_SIZE 1000  // Reasonable upper limit to prevent DoS
#define MIN_MATRIX_SIZE 1

void printMatrix(int **matrix, int N) {
    if (!matrix) return;
    
    for (int i = 0; i < N; i++) {
        if (!matrix[i]) return;
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// Safe memory cleanup function
void freeMatrix(int **matrix, int rows) {
    if (!matrix) return;
    
    for (int i = 0; i < rows; i++) {
        if (matrix[i]) {
            free(matrix[i]);
        }
    }
    free(matrix);
}

// Safe matrix allocation with proper error handling
int** allocateMatrix(int N) {
    // Check for potential overflow in allocation size
    if (N > INT_MAX / sizeof(int*)) {
        return NULL;
    }
    
    int **matrix = (int **)calloc(N, sizeof(int *));
    if (!matrix) {
        return NULL;
    }
    
    for (int i = 0; i < N; i++) {
        matrix[i] = (int *)calloc(N, sizeof(int));
        if (!matrix[i]) {
            // Clean up previously allocated rows
            freeMatrix(matrix, i);
            return NULL;
        }
    }
    
    return matrix;
}

// Safe multiplication with overflow detection
int safe_multiply_add(int a, int b, int c) {
    // Check for multiplication overflow
    if (a != 0 && b != 0) {
        if (a > INT_MAX / b || a < INT_MIN / b) {
            // Overflow would occur, clamp to INT_MAX or INT_MIN
            return (a > 0 && b > 0) ? INT_MAX : INT_MIN;
        }
    }
    
    long long result = (long long)a * b;
    
    // Check for addition overflow
    if (c > 0 && result > INT_MAX - c) {
        return INT_MAX;
    }
    if (c < 0 && result < INT_MIN - c) {
        return INT_MIN;
    }
    
    return (int)(result + c);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        fprintf(stderr, "Where N is matrix size (1-%d)\n", MAX_MATRIX_SIZE);
        return 1;
    }

    // Safe string to integer conversion with validation
    char *endptr;
    errno = 0;
    long N_long = strtol(argv[1], &endptr, 10);
    
    // Validate conversion and range
    if (errno == ERANGE || *endptr != '\0' || endptr == argv[1]) {
        fprintf(stderr, "Error: Invalid number format\n");
        return 1;
    }
    
    if (N_long < MIN_MATRIX_SIZE || N_long > MAX_MATRIX_SIZE) {
        fprintf(stderr, "Error: N must be between %d and %d\n", 
                MIN_MATRIX_SIZE, MAX_MATRIX_SIZE);
        return 1;
    }
    
    int N = (int)N_long;

    // Allocate matrices with proper error checking
    int **A = allocateMatrix(N);
    if (!A) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix A\n");
        return 1;
    }
    
    int **B = allocateMatrix(N);
    if (!B) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix B\n");
        freeMatrix(A, N);
        return 1;
    }
    
    int **C = allocateMatrix(N);
    if (!C) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix C\n");
        freeMatrix(A, N);
        freeMatrix(B, N);
        return 1;
    }

    // Use a better random seed (though for production, consider using a CSPRNG)
    srand((unsigned int)time(NULL) ^ (unsigned int)getpid());

    // Fill matrices A and B with random numbers (limited range to prevent overflow)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;  // Keep small to reduce overflow risk
            B[i][j] = rand() % 100;
        }
    }

    // Matrix multiplication with overflow protection
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] = safe_multiply_add(A[i][k], B[k][j], C[i][j]);
            }
        }
    }

    // Print result matrix
    printf("Result matrix C (A*B):\n");
    printMatrix(C, N);

    // Clean up all allocated memory
    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C, N);

    return 0;
}