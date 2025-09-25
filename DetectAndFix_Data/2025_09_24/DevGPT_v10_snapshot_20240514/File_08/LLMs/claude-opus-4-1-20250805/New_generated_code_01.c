#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#define MAX_NAME_LEN 14  // 15 - 1 for null terminator
#define MAX_RECORDS_PER_CHUNK 100
#define SAFE_STR_COPY(dest, src, size) do { \
    strncpy(dest, src, size - 1); \
    dest[size - 1] = '\0'; \
} while(0)

// Record structure with fixed-size buffers
typedef struct Record {
    char name[15];
    char surname[15];
    char city[15];
    int id;
    char delimiter[2];
} Record;

// CHUNK structure definition (was missing)
typedef struct CHUNK {
    int file_desc;
    int from_BlockId;
    int to_BlockId;
    int recordsInChunk;
    int blocksInChunk;
    Record records[MAX_RECORDS_PER_CHUNK];  // Fixed array of records
} CHUNK;

// CHUNK_Iterator structure (was missing)
typedef struct CHUNK_Iterator {
    int file_desc;
    int current_block;
    int chunk_size;
} CHUNK_Iterator;

// Record Iterator for CHUNK
typedef struct {
    CHUNK chunk;
    int currentBlockId;
    int cursor;
} CHUNK_RecordIterator;

// Stub implementations for missing HP functions
int HP_CreateFile(const char *filename) {
    if (!filename) return -1;
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;
    fclose(fp);
    return 0;
}

int HP_OpenFile(const char *filename, int *file_desc) {
    if (!filename || !file_desc) return -1;
    *file_desc = 1;  // Simplified for demo
    return 0;
}

int HP_CloseFile(int file_desc) {
    return 0;
}

CHUNK_Iterator CHUNK_CreateIterator(int file_desc, int chunk_size) {
    CHUNK_Iterator iter = {file_desc, 0, chunk_size};
    return iter;
}

int CHUNK_GetNext(CHUNK_Iterator *iterator, CHUNK *chunk) {
    if (!iterator || !chunk) return -1;
    
    // Demo implementation - returns one chunk then stops
    if (iterator->current_block > 0) return -1;
    
    chunk->file_desc = iterator->file_desc;
    chunk->from_BlockId = 0;
    chunk->to_BlockId = 1;
    chunk->recordsInChunk = 2;  // Demo: 2 records
    chunk->blocksInChunk = 1;
    
    // Initialize demo records safely
    SAFE_STR_COPY(chunk->records[0].name, "Alice", sizeof(chunk->records[0].name));
    SAFE_STR_COPY(chunk->records[0].surname, "Smith", sizeof(chunk->records[0].surname));
    SAFE_STR_COPY(chunk->records[0].city, "Boston", sizeof(chunk->records[0].city));
    chunk->records[0].id = 101;
    
    SAFE_STR_COPY(chunk->records[1].name, "Bob", sizeof(chunk->records[1].name));
    SAFE_STR_COPY(chunk->records[1].surname, "Johnson", sizeof(chunk->records[1].surname));
    SAFE_STR_COPY(chunk->records[1].city, "Seattle", sizeof(chunk->records[1].city));
    chunk->records[1].id = 102;
    
    iterator->current_block++;
    return 0;
}

// Function declarations
int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record);
int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, const Record *record);
void CHUNK_Print(const CHUNK *chunk);
CHUNK_RecordIterator CHUNK_CreateRecordIterator(const CHUNK *chunk);
int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record);

int main() {
    int file_desc = -1;
    const char *filename = "example_file.txt";
    
    // Check file creation
    if (HP_CreateFile(filename) != 0) {
        fprintf(stderr, "Error: Failed to create file %s\n", filename);
        return 1;
    }
    
    // Check file opening
    if (HP_OpenFile(filename, &file_desc) != 0) {
        fprintf(stderr, "Error: Failed to open file %s\n", filename);
        return 1;
    }
    
    CHUNK_Iterator chunkIterator = CHUNK_CreateIterator(file_desc, 10);
    CHUNK chunk;
    
    while (CHUNK_GetNext(&chunkIterator, &chunk) == 0) {
        CHUNK_Print(&chunk);
        
        Record record;
        CHUNK_RecordIterator recordIterator = CHUNK_CreateRecordIterator(&chunk);
        while (CHUNK_GetNextRecord(&recordIterator, &record) == 0) {
            // Safely print record with length limits
            printf("Name: %.14s, Surname: %.14s, City: %.14s, ID: %d\n", 
                   record.name, record.surname, record.city, record.id);
        }
    }
    
    HP_CloseFile(file_desc);
    return 0;
}

int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record) {
    // Input validation
    if (!chunk || !record) {
        return -1;  // NULL pointer check
    }
    
    // Bounds checking
    if (i < 0 || i >= chunk->recordsInChunk || i >= MAX_RECORDS_PER_CHUNK) {
        return -1;  // Out of bounds
    }
    
    // Safe copy of actual data from chunk
    memcpy(record, &chunk->records[i], sizeof(Record));
    
    // Ensure null termination
    record->name[sizeof(record->name) - 1] = '\0';
    record->surname[sizeof(record->surname) - 1] = '\0';
    record->city[sizeof(record->city) - 1] = '\0';
    record->delimiter[sizeof(record->delimiter) - 1] = '\0';
    
    return 0;  // Success
}

int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, const Record *record) {
    // Input validation
    if (!chunk || !record) {
        return -1;  // NULL pointer check
    }
    
    // Bounds checking
    if (i < 0 || i >= chunk->recordsInChunk || i >= MAX_RECORDS_PER_CHUNK) {
        return -1;  // Out of bounds
    }
    
    // Safe copy with bounds checking
    SAFE_STR_COPY(chunk->records[i].name, record->name, sizeof(chunk->records[i].name));
    SAFE_STR_COPY(chunk->records[i].surname, record->surname, sizeof(chunk->records[i].surname));
    SAFE_STR_COPY(chunk->records[i].city, record->city, sizeof(chunk->records[i].city));
    chunk->records[i].id = record->id;
    SAFE_STR_COPY(chunk->records[i].delimiter, record->delimiter, sizeof(chunk->records[i].delimiter));
    
    return 0;  // Success
}

void CHUNK_Print(const CHUNK *chunk) {
    if (!chunk) {
        fprintf(stderr, "Error: NULL chunk pointer\n");
        return;
    }
    
    printf("File Descriptor: %d\n", chunk->file_desc);
    printf("From Block ID: %d\n", chunk->from_BlockId);
    printf("To Block ID: %d\n", chunk->to_BlockId);
    printf("Records in Chunk: %d\n", chunk->recordsInChunk);
    printf("Blocks in Chunk: %d\n", chunk->blocksInChunk);
    printf("\n");
}

CHUNK_RecordIterator CHUNK_CreateRecordIterator(const CHUNK *chunk) {
    CHUNK_RecordIterator iterator;
    
    if (chunk) {
        iterator.chunk = *chunk;
        iterator.currentBlockId = chunk->from_BlockId;
        iterator.cursor = 0;
    } else {
        // Initialize to safe defaults
        memset(&iterator, 0, sizeof(iterator));
        iterator.cursor = -1;  // Invalid state
    }
    
    return iterator;
}

int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record) {
    // Input validation
    if (!iterator || !record) {
        return -1;  // NULL pointer check
    }
    
    // Check if there are more records to iterate
    if (iterator->cursor < 0 || iterator->cursor >= iterator->chunk.recordsInChunk || 
        iterator->cursor >= MAX_RECORDS_PER_CHUNK) {
        return -1;  // No more records or invalid state
    }
    
    // Safe copy of record data
    memcpy(record, &iterator->chunk.records[iterator->cursor], sizeof(Record));
    
    // Ensure null termination
    record->name[sizeof(record->name) - 1] = '\0';
    record->surname[sizeof(record->surname) - 1] = '\0';
    record->city[sizeof(record->city) - 1] = '\0';
    record->delimiter[sizeof(record->delimiter) - 1] = '\0';
    
    // Move to the next record
    iterator->cursor++;
    
    return 0;  // Success
}