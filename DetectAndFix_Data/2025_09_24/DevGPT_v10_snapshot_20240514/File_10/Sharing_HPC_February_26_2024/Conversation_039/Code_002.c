void multiplyChunk(int **A, int **B, int *resultChunk, int N, int startRow, int endRow) {
    int idx = 0;
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            resultChunk[idx++] = sum;
        }
    }
}