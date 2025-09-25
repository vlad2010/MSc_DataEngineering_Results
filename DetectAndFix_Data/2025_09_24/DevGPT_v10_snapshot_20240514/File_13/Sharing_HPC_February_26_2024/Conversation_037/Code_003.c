void fillMatrixWithZeros(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0; // Fill with zeros
        }
    }
}