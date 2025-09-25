#include <unistd.h>  // For fork(), pipe()
#include <sys/wait.h>  // For wait()

int main(int argc, char *argv[]) {
    int N = 0, verbose = 0, numProcessors = 2; // Default to 2, can be adjusted with command line arguments

    // Process command line arguments, adjust to also get 'numProcessors'
    getCommands(argc, argv, &N, &verbose, &numProcessors);
    
    int **A, **B, **C;
    allocateMatrices(&A, &B, &C, N);
    fillMatrices(A, B, C, N);

    int rowsPerProcessor = N / numProcessors;
    int pipefd[numProcessors][2];  // Array to hold pipe file descriptors for each processor

    for (int i = 0; i < numProcessors; i++) {
        if (pipe(pipefd[i]) == -1) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }

        pid_t pid = fork();
        if (pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) { // Child process
            close(pipefd[i][0]); // Close unused read end

            int startRow = i * rowsPerProcessor;
            int endRow = (i + 1) * rowsPerProcessor;
            if (i == numProcessors - 1) endRow = N; // Make sure the last processor gets any extra rows

            // Child process does its portion of the work
            int *resultChunk = (int *)malloc(rowsPerProcessor * N * sizeof(int));
            multiplyChunk(A, B, resultChunk, N, startRow, endRow);

            // Write the result chunk to the pipe
            write(pipefd[i][1], resultChunk, rowsPerProcessor * N * sizeof(int));
            close(pipefd[i][1]); // Close write end

            free(resultChunk);
            exit(0);
        }
        close(pipefd[i][1]); // Parent closes unused write end
    }

    // Parent process collects the results
    for (int i = 0; i < numProcessors; i++) {
        int startRow = i * rowsPerProcessor;
        int endRow = (i + 1) * rowsPerProcessor;
        if (i == numProcessors - 1) endRow = N;

        // Read the result chunk from the pipe
        read(pipefd[i][0], &C[startRow][0], (endRow - startRow) * N * sizeof(int));
        close(pipefd[i][0]); // Close read end after collecting the data

        wait(NULL); // Wait for child processes to terminate
    }

    if (verbose) {
        printMatrices(A, B, C, N);
        printf("Time Clock to process: %f", cpu_time_used);
    } else {
        printf("%d, %f\n", N, cpu_time_used);
    }

    freeMatrices(A, B, C, N);
    return 0;
}