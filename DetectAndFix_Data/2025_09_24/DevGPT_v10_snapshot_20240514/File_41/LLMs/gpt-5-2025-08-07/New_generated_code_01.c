#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// Define the Record structure
struct Record {
    int  id;
    char name[15];
    char surname[25];
    char address[50];
};

// Safe comparator: bounded byte-wise comparison on fixed-size fields.
// Adds tie-breakers to ensure deterministic ordering.
static int compareRecords(const void *a, const void *b) {
    const struct Record *ra = (const struct Record *)a;
    const struct Record *rb = (const struct Record *)b;

    int c = memcmp(ra->name,    rb->name,    sizeof(ra->name));
    if (c != 0) return c;
    c   = memcmp(ra->surname, rb->surname, sizeof(ra->surname));
    if (c != 0) return c;
    c   = memcmp(ra->address, rb->address, sizeof(ra->address));
    if (c != 0) return c;

    if (ra->id < rb->id) return -1;
    if (ra->id > rb->id) return 1;
    return 0;
}

// Write exactly len bytes unless an error occurs.
// Returns 0 on success, -1 on error with errno set.
static int write_all(int fd, const void *buf, size_t len) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t total = 0;
    while (total < len) {
        ssize_t w = write(fd, p + total, len - total);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) {
            errno = EIO; // Should not happen on regular files unless disk full, etc.
            return -1;
        }
        total += (size_t)w;
    }
    return 0;
}

// Function to "merge" (actually: sort) chunks of records from input file and write to output file
// chunkSize is the number of records to process per chunk.
void merge(int input_FileDesc, int chunkSize, int bWay /*unused*/, int output_FileDesc) {
    (void)bWay; // The current implementation doesn't use multi-way merging.

    if (chunkSize <= 0) {
        fprintf(stderr, "Invalid chunkSize (must be > 0)\n");
        exit(EXIT_FAILURE);
    }

    // Use size_t for sizes to avoid signed overflow/undefined behavior.
    size_t recordsInChunk = (size_t)chunkSize;

    // Check multiplication overflow: recordsInChunk * sizeof(struct Record)
    if (recordsInChunk > SIZE_MAX / sizeof(struct Record)) {
        fprintf(stderr, "chunkSize too large (overflow)\n");
        exit(EXIT_FAILURE);
    }

    size_t bufBytes = recordsInChunk * sizeof(struct Record);

    struct Record *chunk = (struct Record *)malloc(bufBytes);
    if (chunk == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // Read and process chunks until EOF
    for (;;) {
        ssize_t r = read(input_FileDesc, chunk, bufBytes);
        if (r == 0) {
            // EOF
            break;
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue; // retry
            }
            perror("read");
            free(chunk);
            exit(EXIT_FAILURE);
        }

        // r > 0: We got some bytes, which may be less than bufBytes.
        // Validate that we read a whole number of records.
        if ((size_t)r % sizeof(struct Record) != 0) {
            fprintf(stderr, "Input contains a partial record (%zd bytes)\n", r);
            free(chunk);
            exit(EXIT_FAILURE);
        }

        size_t count = (size_t)r / sizeof(struct Record);
        if (count == 0) {
            // Shouldn't happen unless r < sizeof(struct Record), which we reject above.
            continue;
        }

        // Sort only the records that were actually read.
        qsort(chunk, count, sizeof(struct Record), compareRecords);

        // Write exactly the bytes corresponding to 'count' records.
        if (write_all(output_FileDesc, chunk, count * sizeof(struct Record)) != 0) {
            perror("write");
            free(chunk);
            exit(EXIT_FAILURE);
        }
    }

    free(chunk);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <chunk_records>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];

    char *endp = NULL;
    errno = 0;
    long chunk_long = strtol(argv[3], &endp, 10);
    if (errno != 0 || endp == argv[3] || *endp != '\0' || chunk_long <= 0 || chunk_long > INT_MAX) {
        fprintf(stderr, "Invalid chunk_records: %s\n", argv[3]);
        return EXIT_FAILURE;
    }
    int chunkSize = (int)chunk_long;

    // Open input (read-only, no following symlinks if supported), and output securely.
    int in_fd = open(in_path, O_RDONLY
#ifdef O_CLOEXEC
        | O_CLOEXEC
#endif
#ifdef O_NOFOLLOW
        | O_NOFOLLOW
#endif
    );
    if (in_fd == -1) {
        perror("open input");
        return EXIT_FAILURE;
    }

    int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC
#ifdef O_CLOEXEC
        | O_CLOEXEC
#endif
#ifdef O_NOFOLLOW
        | O_NOFOLLOW
#endif
#ifdef O_EXCL
        // Consider O_EXCL if you want to avoid overwriting existing files
        // | O_EXCL
#endif
        , S_IRUSR | S_IWUSR);
    if (out_fd == -1) {
        perror("open output");
        close(in_fd);
        return EXIT_FAILURE;
    }

    // Perform chunked sort ("merge")
    merge(in_fd, chunkSize, /*bWay=*/0, out_fd);

    // Close files
    if (close(in_fd) != 0) {
        perror("close input");
        // continue to close output
    }
    if (close(out_fd) != 0) {
        perror("close output");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}