int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("N value not provided\n");
        return -1;
    }

    int N = atoi(argv[1]); // Converts the command line argument to an integer

    int **A, **B, **C;

    // Allocate memory for all three matrices
    allocateMatrices(N, &A, &B, &C);

    // Your code to fill and work with the matrices goes here

    // Freeing memory for each matrix
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