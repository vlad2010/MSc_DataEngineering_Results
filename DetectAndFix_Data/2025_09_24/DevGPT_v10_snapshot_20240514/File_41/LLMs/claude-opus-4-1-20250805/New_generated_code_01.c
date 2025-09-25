#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>

#define RECORDS_PER_CHUNK 100  // Define a reasonable default
#define MAX_CHUNK_SIZE 10000    // Maximum allowed chunk size for safety

// Define the Record structure
struct Record {
    int id;
    char name[15];
    char surname[25];
    char address[50];
};

// Function to compare two records lexicographically
int compareRecords(const void *a, const void *b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    const struct Record *recA = (const struct Record *)a;
    const struct Record *recB = (const struct Record *)b;
    return strncmp(recA->name, recB->name, sizeof(recA->name) - 1);
}

// Safe multiplication with overflow check
int safe_multiply(int a, int b, int *result) {
    if (a > 0 && b > 0 && a > INT_MAX / b) {
        return -1; // Overflow would occur
    }
    if (a < 0 && b < 0 && a < INT_MAX / b) {
        return -1; // Overflow would occur
    }
    *result = a * b;
    return 0;
}

// Function to merge chunks of records from input file and write to output file
int merge(int input_FileDesc, int chunkSize, int bWay, int output_FileDesc) {
    // Validate input parameters
    if (input_FileDesc < 0 || output_FileDesc < 0) {
        fprintf(stderr, "Invalid file descriptors\n");
        return -1;
    }
    
    if (chunkSize <= 0 || chunkSize > MAX_CHUNK_SIZE) {
        fprintf(stderr, "Invalid chunk size: %d (must be between 1 and %d)\n", 
                chunkSize, MAX_CHUNK_SIZE);
        return -1;
    }
    
    // Calculate the number of records in each chunk with overflow check
    int recordsInChunk;
    if (safe_multiply(chunkSize, RECORDS_PER_CHUNK, &recordsInChunk) != 0) {
        fprintf(stderr, "Chunk size too large, would cause overflow\n");
        return -1;
    }
    
    // Check for allocation size overflow
    size_t alloc_size;
    if (__builtin_umull_overflow(recordsInChunk, sizeof(struct Record), &alloc_size)) {
        fprintf(stderr, "Allocation size would overflow\n");
        return -1;
    }

    // Allocate memory for one chunk of records
    struct Record *chunk = (struct Record *)calloc(recordsInChunk, sizeof(struct Record));
    if (chunk == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        return -1;
    }

    ssize_t bytesRead;
    size_t chunkSizeBytes = recordsInChunk * sizeof(struct Record);
    int totalRecordsProcessed = 0;
    
    // Read and merge chunks until the end of the input file
    while ((bytesRead = read(input_FileDesc, chunk, chunkSizeBytes)) > 0) {
        // Calculate actual number of complete records read
        size_t recordsRead = bytesRead / sizeof(struct Record);
        
        if (recordsRead == 0) {
            break; // No complete records read
        }
        
        // Check for partial record read
        if (bytesRead % sizeof(struct Record) != 0) {
            fprintf(stderr, "Warning: Partial record read, ignoring incomplete data\n");
        }
        
        // Sort only the records that were actually read
        qsort(chunk, recordsRead, sizeof(struct Record), compareRecords);
        
        // Write the sorted chunk to the output file with error checking
        size_t bytesToWrite = recordsRead * sizeof(struct Record);
        ssize_t totalWritten = 0;
        
        while (totalWritten < bytesToWrite) {
            ssize_t written = write(output_FileDesc, 
                                   ((char*)chunk) + totalWritten, 
                                   bytesToWrite - totalWritten);
            if (written < 0) {
                if (errno == EINTR) {
                    continue; // Retry if interrupted
                }
                perror("Write error");
                free(chunk);
                return -1;
            }
            totalWritten += written;
        }
        
        totalRecordsProcessed += recordsRead;
        
        // Clear the buffer for next iteration (prevent information leakage)
        memset(chunk, 0, chunkSizeBytes);
    }
    
    if (bytesRead < 0) {
        perror("Read error");
        free(chunk);
        return -1;
    }

    // Free allocated memory
    free(chunk);
    
    printf("Successfully processed %d records\n", totalRecordsProcessed);
    return 0;
}

int main() {
    // Example usage with proper initialization
    int input_FileDesc = -1, output_FileDesc = -1;
    int chunkSize = 10;  // Set a reasonable chunk size
    int bWay = 2;         // Set bWay value (though not used in current implementation)
    int ret = 0;
    
    // Open input and output files with proper error checking
    input_FileDesc = open("input_file", O_RDONLY);
    if (input_FileDesc == -1) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }
    
    output_FileDesc = open("output_file", O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (output_FileDesc == -1) {
        perror("Error opening output file");
        close(input_FileDesc);
        return EXIT_FAILURE;
    }

    // Call the merge function with error checking
    if (merge(input_FileDesc, chunkSize, bWay, output_FileDesc) != 0) {
        fprintf(stderr, "Merge operation failed\n");
        ret = EXIT_FAILURE;
    }

    // Close files
    if (close(input_FileDesc) != 0) {
        perror("Error closing input file");
        ret = EXIT_FAILURE;
    }
    
    if (close(output_FileDesc) != 0) {
        perror("Error closing output file");
        ret = EXIT_FAILURE;
    }

    return ret;
}