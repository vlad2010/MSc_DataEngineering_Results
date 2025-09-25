void allocateMatrices(int ***A, int ***B, int ***C, int N){
    *A = (int **)malloc(N * sizeof(int *));
    *B = (int **)malloc(N * sizeof(int *));
    *C = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        (*A)[i] = (int *)malloc(N * sizeof(int));
        (*B)[i] = (int *)malloc(N * sizeof(int));
        (*C)[i] = (int *)malloc(N * sizeof(int)); // Use calloc for C to initialize to 0
    }
}