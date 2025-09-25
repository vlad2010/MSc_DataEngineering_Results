#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX_RECORDS_PER_CHUNK 100
#define MAX_NAME_LENGTH 14
#define MAX_SURNAME_LENGTH 24
#define MAX_ADDRESS_LENGTH 49

// Define the Record structure with proper bounds
struct Record {
    int id;
    char name[MAX_NAME_LENGTH + 1];      // +1 for null terminator
    char surname[MAX_SURNAME_LENGTH + 1]; // +1 for null terminator
    char address[MAX_ADDRESS_LENGTH + 1]; // +1 for null terminator
};

// Define the CHUNK structure with fixed array size
struct CHUNK {
    int file_desc;
    int from_BlockId;
    int to_BlockId;
    int recordsInChunk;
    int blocksInChunk;
    struct Record records[MAX_RECORDS_PER_CHUNK];
};

// Safe string comparison helper function
int safe_strcmp(const char* s1, const char* s2, size_t max_len) {
    if (s1 == NULL || s2 == NULL) {
        return 0;
    }
    return strncmp(s1, s2, max_len);
}

// Function to compare two records lexicographically (fixed logic)
bool shouldSwap(const struct Record* rec1, const struct Record* rec2) {
    if (rec1 == NULL || rec2 == NULL) {
        return false;
    }
    
    int nameComparison = safe_strcmp(rec1->name, rec2->name, MAX_NAME_LENGTH);

    if (nameComparison > 0) {
        // rec1's name comes after rec2's name - should swap for ascending order
        return true;
    } else if (nameComparison == 0) {
        // Names are equal, compare surnames
        int surnameComparison = safe_strcmp(rec1->surname, rec2->surname, MAX_SURNAME_LENGTH);

        if (surnameComparison > 0) {
            // rec1's surname comes after rec2's surname - should swap for ascending order
            return true;
        }
    }

    // No need to swap
    return false;
}

// Function to perform bubble sort on records within a chunk with bounds checking
void sort_Chunk(struct CHUNK* chunk) {
    if (chunk == NULL) {
        return;
    }
    
    // Validate recordsInChunk is within bounds
    if (chunk->recordsInChunk < 0 || chunk->recordsInChunk > MAX_RECORDS_PER_CHUNK) {
        fprintf(stderr, "Error: Invalid number of records in chunk: %d\n", chunk->recordsInChunk);
        return;
    }
    
    int i, j;
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

// Safe string copy helper function
void safe_strcpy(char* dest, const char* src, size_t dest_size) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return;
    }
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';  // Ensure null termination
}

// Function to safely initialize a record
void init_record(struct Record* rec, int id, const char* name, const char* surname, const char* address) {
    if (rec == NULL) {
        return;
    }
    
    rec->id = id;
    safe_strcpy(rec->name, name ? name : "", sizeof(rec->name));
    safe_strcpy(rec->surname, surname ? surname : "", sizeof(rec->surname));
    safe_strcpy(rec->address, address ? address : "", sizeof(rec->address));
}

int main() {
    // Properly initialize the chunk structure
    struct CHUNK chunk = {0};  // Zero-initialize all fields
    
    // Initialize the chunk with some sample records
    chunk.file_desc = 1;
    chunk.from_BlockId = 0;
    chunk.to_BlockId = 10;
    chunk.blocksInChunk = 1;
    chunk.recordsInChunk = 3;  // Set the actual number of records
    
    // Safely populate records with sample data
    init_record(&chunk.records[0], 1, "John", "Doe", "123 Main St");
    init_record(&chunk.records[1], 2, "Alice", "Smith", "456 Oak Ave");
    init_record(&chunk.records[2], 3, "Alice", "Johnson", "789 Pine Rd");
    
    // Call the sort_Chunk function to sort the records in the chunk
    sort_Chunk(&chunk);
    
    // Display the sorted records with bounds checking
    printf("Sorted records:\n");
    for (int i = 0; i < chunk.recordsInChunk && i < MAX_RECORDS_PER_CHUNK; i++) {
        printf("Record %d: %s %s (ID: %d, Address: %s)\n", 
               i + 1, 
               chunk.records[i].name, 
               chunk.records[i].surname,
               chunk.records[i].id,
               chunk.records[i].address);
    }
    
    return 0;
}