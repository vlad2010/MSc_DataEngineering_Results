#include <stdio.h>
#include <stdlib.h>

// Define the Record structure
struct Record {
    int id;
    char name[15];
    char surname[25];
    char address[50];
};

// Function to compare two records lexicographically
int compareRecords(const void *a, const void *b) {
    return strcmp(((struct Record *)a)->name, ((struct Record *)b)->name);
}

// Function to merge chunks of records from input file and write to output file
void merge(int input_FileDesc, int chunkSize, int bWay, int output_FileDesc) {
    // Calculate the number of records in each chunk
    int recordsInChunk = chunkSize * SIZE;

    // Allocate memory for one chunk of records
    struct Record *chunk = (struct Record *)malloc(recordsInChunk * sizeof(struct Record));
    if (chunk == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Read and merge chunks until the end of the input file
    while (read(input_FileDesc, chunk, recordsInChunk * sizeof(struct Record)) > 0) {
        // Sort the chunk using qsort and the previously defined compareRecords function
        qsort(chunk, recordsInChunk, sizeof(struct Record), compareRecords);

        // Write the sorted chunk to the output file
        write(output_FileDesc, chunk, recordsInChunk * sizeof(struct Record));
    }

    // Free allocated memory
    free(chunk);
}

int main() {
    // Example usage
    int input_FileDesc, output_FileDesc;
    int chunkSize = /* Set your chunk size */;
    int bWay = /* Set your bWay */;
    
    // Open input and output files (you need to implement this part)
    // input_FileDesc = open("input_file", O_RDONLY);
    // output_FileDesc = open("output_file", O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

    // Check for file open errors (you need to implement this part)
    // if (input_FileDesc == -1 || output_FileDesc == -1) {
    //     perror("Error opening file");
    //     exit(EXIT_FAILURE);
    // }

    // Call the merge function
    merge(input_FileDesc, chunkSize, bWay, output_FileDesc);

    // Close files (you need to implement this part)
    // close(input_FileDesc);
    // close(output_FileDesc);

    return 0;
}