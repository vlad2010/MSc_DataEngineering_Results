#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_RECORDS_IN_CHUNK 100

// Define the Record structure
struct Record {
    int id;
    char name[15];
    char surname[25];
    char address[50];
};

// Define the CHUNK structure
struct CHUNK {
    int file_desc;
    int from_BlockId;
    int to_BlockId;
    int recordsInChunk;
    int blocksInChunk;
    struct Record records[MAX_RECORDS_IN_CHUNK];
};

// Function to compare two records lexicographically
bool shouldSwap(struct Record* rec1, struct Record* rec2) {
    int nameComparison = strcmp(rec1->name, rec2->name);

    if (nameComparison < 0) {
        // rec1's name comes before rec2's name
        return true;
    } else if (nameComparison == 0) {
        // Names are equal, compare surnames
        int surnameComparison = strcmp(rec1->surname, rec2->surname);

        if (surnameComparison < 0) {
            // rec1's surname comes before rec2's surname
            return true;
        }
    }

    // No need to swap
    return false;
}

// Function to perform bubble sort on records within a chunk
void sort_Chunk(struct CHUNK* chunk) {
    int i, j;
    // Defensive: Ensure recordsInChunk is within bounds
    if (chunk->recordsInChunk < 0 || chunk->recordsInChunk > MAX_RECORDS_IN_CHUNK) {
        fprintf(stderr, "Error: recordsInChunk out of bounds\n");
        return;
    }
    for (i = 0; i < chunk->recordsInChunk - 1; i++) {
        for (j = 0; j < chunk->recordsInChunk - i - 1; j++) {
            // Compare adjacent records and swap if needed
            if (shouldSwap(&chunk->records[j], &chunk->records[j + 1])) {
                // Swap records
                struct Record temp = chunk->records[j];
                chunk->records[j] = chunk->records[j + 1];
                chunk->records[j + 1] = temp;
            }
        }
    }
}

int main() {
    // Example usage
    struct CHUNK chunk = {0};
    chunk.recordsInChunk = 3; // Example: 3 records

    // Initialize records safely
    chunk.records[0].id = 1;
    strncpy(chunk.records[0].name, "Alice", sizeof(chunk.records[0].name) - 1);
    chunk.records[0].name[sizeof(chunk.records[0].name) - 1] = '\0';
    strncpy(chunk.records[0].surname, "Smith", sizeof(chunk.records[0].surname) - 1);
    chunk.records[0].surname[sizeof(chunk.records[0].surname) - 1] = '\0';

    chunk.records[1].id = 2;
    strncpy(chunk.records[1].name, "Bob", sizeof(chunk.records[1].name) - 1);
    chunk.records[1].name[sizeof(chunk.records[1].name) - 1] = '\0';
    strncpy(chunk.records[1].surname, "Jones", sizeof(chunk.records[1].surname) - 1);
    chunk.records[1].surname[sizeof(chunk.records[1].surname) - 1] = '\0';

    chunk.records[2].id = 3;
    strncpy(chunk.records[2].name, "Charlie", sizeof(chunk.records[2].name) - 1);
    chunk.records[2].name[sizeof(chunk.records[2].name) - 1] = '\0';
    strncpy(chunk.records[2].surname, "Brown", sizeof(chunk.records[2].surname) - 1);
    chunk.records[2].surname[sizeof(chunk.records[2].surname) - 1] = '\0';

    // Call the sort_Chunk function to sort the records in the chunk
    sort_Chunk(&chunk);

    // Display the sorted records (for demonstration purposes)
    for (int i = 0; i < chunk.recordsInChunk; i++) {
        printf("Record %d: %s %s\n", i + 1, chunk.records[i].name, chunk.records[i].surname);
    }

    return 0;
}