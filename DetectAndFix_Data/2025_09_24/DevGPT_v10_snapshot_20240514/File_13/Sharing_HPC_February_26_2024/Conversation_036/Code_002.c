int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("N value not provided\n");
        return -1;
    }

    int N = atoi(argv[1]); // Converts the command line argument to an integer

    int **A, **B, **C;

    // Memory allocation
    allocateMatrices(&A, &B, &C, N);

    // Filling matrices
    fillMatrices(A, B, C, N);

    // Don't forget to free the allocated memory and add the rest of your program logic here

    return 0;
}