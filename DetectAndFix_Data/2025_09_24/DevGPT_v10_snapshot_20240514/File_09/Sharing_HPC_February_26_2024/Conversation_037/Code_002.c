void fillMatrixWithRandomValues(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 100; // Fill with random values between 0 and 99
        }
    }
}