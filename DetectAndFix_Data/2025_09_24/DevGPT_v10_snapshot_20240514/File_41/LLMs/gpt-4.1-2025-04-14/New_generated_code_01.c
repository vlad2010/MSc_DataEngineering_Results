#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // For strcmp
#include <unistd.h>     // For read, write, close
#include <fcntl.h>      // For open
#include <errno.h>      // For errno

// Define the Record structure
struct Record {
    int id;
    char name[15];
    char surname[25];
    char address[50];
};

// Safe string compare for fixed-size char arrays
int safe_strncmp(const char *a, const char *b, size_t n) {
    // Ensure null-termination for comparison
    char a_buf[64], b_buf[64];
    strncpy(a_buf, a, n);
    a_buf[n-1] = '\0';
    strncpy(b_buf, b, n);
    b_buf[n-1] = '\0';
    return strcmp(a_buf, b_buf);
}

// Function to compare two records lexicographically
int compareRecords(const void *a, const void *b) {
    // Use safe_strncmp to avoid buffer over-read
    return safe_strncmp(((const struct Record *)a)->name, ((const struct Record *)b)->name, sizeof(((struct Record *)a)->name));
}

// Function to merge chunks of records from input file and write to output file
void merge(int input_FileDesc, int chunkSize, int bWay, int output_FileDesc) {
    // Define the number of records per chunk
    // For demonstration, assume chunkSize is the number of records per chunk
    if (chunkSize <= 0) {
        fprintf(stderr, "Invalid chunk size\n");
        exit(EXIT_FAILURE);
    }

    size_t recordsInChunk = (size_t)chunkSize;

    // Check for integer overflow in allocation
    if (recordsInChunk > SIZE_MAX / sizeof(struct Record)) {
        fprintf(stderr, "Chunk size too large\n");
        exit(EXIT_FAILURE);
    }

    struct Record *chunk = (struct Record *)malloc(recordsInChunk * sizeof(struct Record));
    if (chunk == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    ssize_t bytesRead;
    size_t recordSize = sizeof(struct Record);

    while ((bytesRead = read(input_FileDesc, chunk, recordsInChunk * recordSize)) > 0) {
        size_t recordsRead = bytesRead / recordSize;

        // Sort only the records actually read
        qsort(chunk, recordsRead, recordSize, compareRecords);

        // Write sorted records, handle partial writes
        size_t totalWritten = 0;
        while (totalWritten < recordsRead) {
            ssize_t bytesWritten = write(output_FileDesc, (char *)chunk + totalWritten * recordSize, (recordsRead - totalWritten) * recordSize);
            if (bytesWritten < 0) {
                perror("Error writing to output file");
                free(chunk);
                exit(EXIT_FAILURE);
            }
            totalWritten += bytesWritten / recordSize;
        }
    }

    if (bytesRead < 0) {
        perror("Error reading from input file");
        free(chunk);
        exit(EXIT_FAILURE);
    }

    free(chunk);
}

int main() {
    int input_FileDesc, output_FileDesc;
    int chunkSize = 100; // Example chunk size
    int bWay = 2;        // Example value

    // Open input and output files
    input_FileDesc = open("input_file", O_RDONLY);
    if (input_FileDesc == -1) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }
    output_FileDesc = open("output_file", O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (output_FileDesc == -1) {
        perror("Error opening output file");
        close(input_FileDesc);
        exit(EXIT_FAILURE);
    }

    // Call the merge function
    merge(input_FileDesc, chunkSize, bWay, output_FileDesc);

    // Close files
    if (close(input_FileDesc) == -1) {
        perror("Error closing input file");
    }
    if (close(output_FileDesc) == -1) {
        perror("Error closing output file");
    }

    return 0;
}