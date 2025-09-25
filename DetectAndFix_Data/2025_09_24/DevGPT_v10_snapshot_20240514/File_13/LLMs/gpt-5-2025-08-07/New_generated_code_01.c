#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <limits.h>

#define MAX_PROCESSORS 4
// Cap total matrix memory to mitigate DoS on shared memory.
// Adjust conservatively per environment (here: 512 MiB).
#define MAX_MATRIX_BYTES (512ULL * 1024ULL * 1024ULL)

static bool calc_matrix_bytes(size_t n, size_t elem_size, size_t *out_bytes) {
    if (n == 0) return false;
    if (n > SIZE_MAX / n) return false;                 // n*n overflow
    size_t n2 = n * n;
    if (n2 > SIZE_MAX / elem_size) return false;        // n2*elem_size overflow
    *out_bytes = n2 * elem_size;
    return true;
}

static bool parse_size_t(const char *s, size_t *out) {
    if (!s || !*s) return false;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;
    if (v == 0) return false;                           // zero is not valid N
    if (v > SIZE_MAX) return false;
    *out = (size_t)v;
    return true;
}

static bool parse_uint(const char *s, unsigned *out) {
    if (!s || !*s) return false;
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;
    if (v == 0 || v > UINT_MAX) return false;           // require >= 1
    *out = (unsigned)v;
    return true;
}

static void allocateSharedMatrix(int *shmid, size_t bytes) {
    // Restrictive permissions; umask may further reduce.
    int id = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
    if (id < 0) {
        perror("shmget");
        exit(1);
    }
    *shmid = id;
}

static int** attachMatrix(int shmid, size_t N) {
    int *data = (int *)shmat(shmid, NULL, 0);
    if (data == (int *)-1) {
        perror("shmat");
        exit(1);
    }
    int **matrix = (int **)malloc(N * sizeof(int *));
    if (!matrix) {
        perror("malloc");
        // Detach the segment we just attached before exit.
        if (shmdt(data) == -1) perror("shmdt");
        exit(1);
    }
    for (size_t i = 0; i < N; i++) {
        matrix[i] = data + N * i;
    }
    return matrix;
}

static void detachMatrix(int **matrix) {
    if (!matrix) return;
    if (matrix[0]) {
        // matrix[0] points to the start of the shared segment.
        if (shmdt(matrix[0]) == -1) {
            perror("shmdt");
        }
    }
    free(matrix);
}

static void fillMatrix(int **matrix, size_t N, int isRandom) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            matrix[i][j] = isRandom ? (rand() % 100) : 0;
        }
    }
}

static void multiplyMatrixChunk(int **A, int **B, int **C, size_t N, size_t startRow, size_t endRow) {
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t j = 0; j < N; j++) {
            long long acc = C[i][j]; // avoid intermediate overflow on add (still 32-bit mul)
            for (size_t k = 0; k < N; k++) {
                acc += (long long)A[i][k] * (long long)B[k][j];
            }
            // Cast back to int; for large N and values, this may overflow mathematically,
            // but the type is preserved with explicit cast.
            C[i][j] = (int)acc;
        }
    }
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <N> [-p num_processors]\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    size_t N = 0;
    if (!parse_size_t(argv[1], &N)) {
        fprintf(stderr, "Invalid N: %s\n", argv[1]);
        exit(1);
    }

    // Optional: parse -p
    unsigned numProcessors = MAX_PROCESSORS;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            unsigned p = 0;
            if (!parse_uint(argv[++i], &p)) {
                fprintf(stderr, "Invalid number of processors: %s\n", argv[i]);
                exit(1);
            }
            if (p == 0 || p > MAX_PROCESSORS) {
                fprintf(stderr, "num_processors must be in [1, %d]\n", MAX_PROCESSORS);
                exit(1);
            }
            numProcessors = p;
        } else {
            usage(argv[0]);
            exit(1);
        }
    }

    // Prevent using more processes than rows (avoid waste/overhead).
    if (N < numProcessors) numProcessors = (unsigned)N;
    if (numProcessors == 0) numProcessors = 1;

    // Compute safe size and limit total memory.
    size_t bytes = 0;
    if (!calc_matrix_bytes(N, sizeof(int), &bytes)) {
        fprintf(stderr, "Matrix size overflow for N=%zu\n", N);
        exit(1);
    }
    // We need 3 matrices of this size.
    if (bytes > MAX_MATRIX_BYTES / 3) {
        fprintf(stderr, "Requested matrix too large: N=%zu exceeds resource limit.\n", N);
        exit(1);
    }

    srand((unsigned)time(NULL));

    int shmidA = -1, shmidB = -1, shmidC = -1;
    allocateSharedMatrix(&shmidA, bytes);
    allocateSharedMatrix(&shmidB, bytes);
    allocateSharedMatrix(&shmidC, bytes);

    int **A = attachMatrix(shmidA, N);
    int **B = attachMatrix(shmidB, N);
    int **C = attachMatrix(shmidC, N);

    // Defense-in-depth: prevent other (non-child) processes from attaching.
    // Children already inherit these attachments via fork().
    if (shmctl(shmidA, IPC_RMID, NULL) == -1) perror("shmctl IPC_RMID A");
    if (shmctl(shmidB, IPC_RMID, NULL) == -1) perror("shmctl IPC_RMID B");
    if (shmctl(shmidC, IPC_RMID, NULL) == -1) perror("shmctl IPC_RMID C");

    fillMatrix(A, N, 1);
    fillMatrix(B, N, 1);
    fillMatrix(C, N, 0);

    // Compute row distribution safely (ceiling division).
    size_t rowsPerProcessor = (N + numProcessors - 1) / numProcessors;

    // Fork workers
    unsigned launched = 0;
    for (unsigned p = 0; p < numProcessors; p++) {
        size_t startRow = p * rowsPerProcessor;
        if (startRow >= N) break;
        size_t endRow = startRow + rowsPerProcessor;
        if (endRow > N) endRow = N;

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            // Stop launching more; proceed to wait for already launched children.
            break;
        } else if (pid == 0) {
            // Child
            multiplyMatrixChunk(A, B, C, N, startRow, endRow);
            detachMatrix(A);
            detachMatrix(B);
            detachMatrix(C);
            _exit(0);
        } else {
            launched++;
        }
    }

    // Wait for children
    int status;
    while (launched > 0) {
        pid_t w = wait(&status);
        if (w == -1) {
            if (errno == EINTR) continue;
            if (errno == ECHILD) break;
        } else {
            launched--;
        }
    }

    // Optionally print C here

    // Cleanup: segments were marked for deletion; just detach.
    detachMatrix(A);
    detachMatrix(B);
    detachMatrix(C);

    return 0;
}