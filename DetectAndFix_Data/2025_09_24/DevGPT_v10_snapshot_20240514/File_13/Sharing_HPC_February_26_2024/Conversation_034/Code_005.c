int main() {
    // Allocate and initialize matrices A, B, and C
    int **A, **B, **C;
    // Matrix allocation and initialization code

    for (int i = 0; i < N; i++) {
        pid_t pid = fork();
        if (pid == 0) {  // Child process
            // Perform multiplication for a chunk/row of the matrix
            // Ensure that each child writes to a unique part of matrix C
            exit(0);
        }
    }

    // Parent process waits for all children to complete
    while (wait(NULL) > 0);

    // Use results stored in matrix C
    // No need for shmdt or shmctl since we're not using explicit shared memory

    return 0;
}