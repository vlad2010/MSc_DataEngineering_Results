#include <stdio.h>
#include <string.h>

#define NAME_SIZE 15
#define SURNAME_SIZE 15
#define CITY_SIZE 15
#define DELIMITER_SIZE 2

typedef struct Record {
    char name[NAME_SIZE];
    char surname[SURNAME_SIZE];
    char city[CITY_SIZE];
    int id;
    char delimiter[DELIMITER_SIZE];
} Record;

// Assume CHUNK and related functions are defined elsewhere
typedef struct {
    int file_desc;
    int from_BlockId;
    int to_BlockId;
    int recordsInChunk;
    int blocksInChunk;
    Record *records; // Assuming this is a pointer to an array of Record
} CHUNK;

int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record);
int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, Record record);
void CHUNK_Print(CHUNK chunk);

// Corrected Record Iterator for CHUNK
typedef struct {
    CHUNK chunk;
    int currentBlockId;
    int cursor;
} CHUNK_RecordIterator;

CHUNK_RecordIterator CHUNK_CreateRecordIterator(CHUNK *chunk);
int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record);

int main() {
    // Example usage:
    int file_desc;
    char *filename = "example_file.txt";
    HP_CreateFile(filename);
    HP_OpenFile(filename, &file_desc);

    CHUNK_Iterator chunkIterator = CHUNK_CreateIterator(file_desc, 10);
    CHUNK chunk;

    while (CHUNK_GetNext(&chunkIterator, &chunk) == 0) {
        CHUNK_Print(chunk);

        // Retrieve and update records in the CHUNK
        Record record;
        CHUNK_RecordIterator recordIterator = CHUNK_CreateRecordIterator(&chunk);
        while (CHUNK_GetNextRecord(&recordIterator, &record) == 0) {
            // Process each record
            // Example: Print the record
            printf("Name: %.*s, Surname: %.*s, City: %.*s, ID: %d\n",
                NAME_SIZE-1, record.name,
                SURNAME_SIZE-1, record.surname,
                CITY_SIZE-1, record.city,
                record.id);
        }
    }

    HP_CloseFile(file_desc);

    return 0;
}

int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record) {
    // Check if the index is within bounds
    if (i < 0 || i >= chunk->recordsInChunk) {
        return -1; // Out of bounds
    }

    // Populate the record based on the index in your data structure
    // Example: Assuming Record has fields name, surname, city, id
    // You may need to adjust this based on your actual data structure
    snprintf(record->name, NAME_SIZE, "%s", "John");
    snprintf(record->surname, SURNAME_SIZE, "%s", "Doe");
    snprintf(record->city, CITY_SIZE, "%s", "New York");
    record->id = 123;

    return 0; // Success
}

int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, Record record) {
    // Check if the index is within bounds
    if (i < 0 || i >= chunk->recordsInChunk) {
        return -1; // Out of bounds
    }

    // Update the ith record based on your logic
    // Example: Assuming Record has fields name, surname, city, id
    // You may need to adjust this based on your actual data structure
    snprintf(chunk->records[i].name, NAME_SIZE, "%s", record.name);
    snprintf(chunk->records[i].surname, SURNAME_SIZE, "%s", record.surname);
    snprintf(chunk->records[i].city, CITY_SIZE, "%s", record.city);
    chunk->records[i].id = record.id;

    return 0; // Success
}

void CHUNK_Print(CHUNK chunk) {
    printf("File Descriptor: %d\n", chunk.file_desc);
    printf("From Block ID: %d\n", chunk.from_BlockId);
    printf("To Block ID: %d\n", chunk.to_BlockId);
    printf("Records in Chunk: %d\n", chunk.recordsInChunk);
    printf("Blocks in Chunk: %d\n", chunk.blocksInChunk);
    printf("\n");
}

CHUNK_RecordIterator CHUNK_CreateRecordIterator(CHUNK *chunk) {
    CHUNK_RecordIterator iterator;
    iterator.chunk = *chunk;
    iterator.currentBlockId = chunk->from_BlockId;
    iterator.cursor = 0; // Start from the first record in the first block
    return iterator;
}

int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record) {
    // Check if there are more records to iterate
    if (iterator->cursor >= iterator->chunk.recordsInChunk) {
        return -1; // No more records
    }

    // Retrieve the next record based on your data structure
    // Example: Assuming Record has fields name, surname, city, id
    // You may need to adjust this based on your actual data structure
    snprintf(record->name, NAME_SIZE, "%s", iterator->chunk.records[iterator->cursor].name);
    snprintf(record->surname, SURNAME_SIZE, "%s", iterator->chunk.records[iterator->cursor].surname);
    snprintf(record->city, CITY_SIZE, "%s", iterator->chunk.records[iterator->cursor].city);
    record->id = iterator->chunk.records[iterator->cursor].id;

    // Move to the next record
    iterator->cursor++;

    return 0; // Success
}