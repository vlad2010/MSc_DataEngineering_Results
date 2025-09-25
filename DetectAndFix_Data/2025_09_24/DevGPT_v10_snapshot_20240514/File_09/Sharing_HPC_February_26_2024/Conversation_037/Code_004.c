int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]); // Matrix size from command line
    srand(time(NULL)); // Seed for random number generation

    // Allocate matrices
    int** A = allocateMatrix(N);
    int** B = allocateMatrix(N);
    int** C = allocateMatrix(N); // Assuming C is the result matrix

    // Fill matrices A and B with random values
    fillMatrixWithRandomValues(A, N);
    fillMatrixWithRandomValues(B, N);

    // Optionally, fill matrix C with zeros if needed
    fillMatrixWithZeros(C, N);

    // Your matrix multiplication logic and any other processing goes here

    // Don't forget to free the allocated memory for each matrix to avoid memory leaks
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}