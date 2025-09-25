#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>

/*
 Security fixes included:
 - Replaced unsafe strcpy with bounded copies that always NUL-terminate (CWE-120/121/787).
 - Increased delimiter buffer to include space for NUL (CWE-131).
 - Added NULL checks to prevent deref (CWE-476).
 - Added bounds validation against MAX_RECORDS (CWE-787/119).
 - Checked return values of file operations (CWE-252).
*/

// Configurable limits for compile-time safety
#define NAME_LEN     15
#define SURNAME_LEN  15
#define CITY_LEN     15
#define DELIM_LEN    3      // 2 characters + NUL
#define MAX_RECORDS  128

// Safe bounded copy helper: always NUL-terminates destination.
static void bounded_copy(char *dst, size_t dstsz, const char *src) {
    if (!dst || dstsz == 0) return;
    if (!src) { dst[0] = '\0'; return; }
    size_t n = strnlen(src, dstsz - 1);
    memcpy(dst, src, n);
    dst[n] = '\0';
}

// Record type
typedef struct Record {
    char name[NAME_LEN];
    char surname[SURNAME_LEN];
    char city[CITY_LEN];
    int  id;
    char delimiter[DELIM_LEN]; // fixed size (CWE-131)
} Record;

// Minimal CHUNK and iterator mock-up for demonstration purposes
typedef struct CHUNK {
    int file_desc;
    int from_BlockId;
    int to_BlockId;
    int recordsInChunk;
    int blocksInChunk;
    Record records[MAX_RECORDS];
} CHUNK;

typedef struct {
    int file_desc;
    int limit;
    int current;
} CHUNK_Iterator;

CHUNK_Iterator CHUNK_CreateIterator(int file_desc, int limit) {
    CHUNK_Iterator it = {0};
    it.file_desc = file_desc;
    it.limit = (limit > 0) ? limit : 0;
    it.current = 0;
    return it;
}

// Returns 0 when a chunk is produced, -1 when iteration ends or on error.
int CHUNK_GetNext(CHUNK_Iterator *it, CHUNK *out) {
    if (!it || !out) return -1; // CWE-476
    if (it->current >= it->limit) return -1;

    // Demo: populate a single chunk safely
    memset(out, 0, sizeof(*out));
    out->file_desc = it->file_desc;
    out->from_BlockId = it->current * 100;
    out->to_BlockId = out->from_BlockId + 9;
    out->blocksInChunk = 10;

    out->recordsInChunk = 2; // Ensure <= MAX_RECORDS
    bounded_copy(out->records[0].name, sizeof(out->records[0].name), "John");
    bounded_copy(out->records[0].surname, sizeof(out->records[0].surname), "Doe");
    bounded_copy(out->records[0].city, sizeof(out->records[0].city), "New York");
    out->records[0].id = 123;
    bounded_copy(out->records[0].delimiter, sizeof(out->records[0].delimiter), ",");

    bounded_copy(out->records[1].name, sizeof(out->records[1].name), "Alice");
    bounded_copy(out->records[1].surname, sizeof(out->records[1].surname), "Smith");
    bounded_copy(out->records[1].city, sizeof(out->records[1].city), "Paris");
    out->records[1].id = 456;
    bounded_copy(out->records[1].delimiter, sizeof(out->records[1].delimiter), ";");

    it->current++;
    return 0;
}

// Minimal file API stubs for demo
int HP_CreateFile(const char *filename) {
    if (!filename) return -1;
    // Simulate success.
    return 0;
}
int HP_OpenFile(const char *filename, int *file_desc) {
    if (!filename || !file_desc) return -1;
    *file_desc = 3; // demo fd
    return 0;
}
int HP_CloseFile(int file_desc) {
    (void)file_desc;
    return 0;
}

// API from the original snippet
int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record);
int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, Record record);
void CHUNK_Print(const CHUNK chunk);

// Record Iterator for CHUNK
typedef struct {
    CHUNK *chunk;
    int currentRecord;
} CHUNK_RecordIterator;

CHUNK_RecordIterator CHUNK_CreateRecordIterator(CHUNK *chunk);
int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record);

// Implementations with security fixes

int CHUNK_GetIthRecordInChunk(CHUNK *chunk, int i, Record *record) {
    if (!chunk || !record) return -1; // CWE-476
    if (i < 0 || i >= chunk->recordsInChunk) return -1;
    if (chunk->recordsInChunk < 0 || chunk->recordsInChunk > MAX_RECORDS) return -1; // CWE-787

    // Safe bounded copies (CWE-120/121/787)
    bounded_copy(record->name, sizeof(record->name), chunk->records[i].name);
    bounded_copy(record->surname, sizeof(record->surname), chunk->records[i].surname);
    bounded_copy(record->city, sizeof(record->city), chunk->records[i].city);
    bounded_copy(record->delimiter, sizeof(record->delimiter), chunk->records[i].delimiter);
    record->id = chunk->records[i].id;

    return 0;
}

int CHUNK_UpdateIthRecord(CHUNK *chunk, int i, Record record) {
    if (!chunk) return -1; // CWE-476
    if (i < 0 || i >= chunk->recordsInChunk) return -1;
    if (chunk->recordsInChunk < 0 || chunk->recordsInChunk > MAX_RECORDS) return -1; // CWE-787

    // Safe bounded copies (CWE-120/121/787)
    bounded_copy(chunk->records[i].name, sizeof(chunk->records[i].name), record.name);
    bounded_copy(chunk->records[i].surname, sizeof(chunk->records[i].surname), record.surname);
    bounded_copy(chunk->records[i].city, sizeof(chunk->records[i].city), record.city);
    bounded_copy(chunk->records[i].delimiter, sizeof(chunk->records[i].delimiter), record.delimiter);
    chunk->records[i].id = record.id;

    return 0;
}

void CHUNK_Print(const CHUNK chunk) {
    // Simple info print (format string is constant -> no CWE-134)
    printf("File Descriptor: %d\n", chunk.file_desc);
    printf("From Block ID: %d\n", chunk.from_BlockId);
    printf("To Block ID: %d\n", chunk.to_BlockId);
    printf("Records in Chunk: %d\n", chunk.recordsInChunk);
    printf("Blocks in Chunk: %d\n", chunk.blocksInChunk);
    printf("\n");
}

CHUNK_RecordIterator CHUNK_CreateRecordIterator(CHUNK *chunk) {
    CHUNK_RecordIterator iterator;
    iterator.chunk = chunk;
    iterator.currentRecord = 0;
    return iterator;
}

int CHUNK_GetNextRecord(CHUNK_RecordIterator *iterator, Record *record) {
    if (!iterator || !iterator->chunk || !record) return -1; // CWE-476

    if (iterator->chunk->recordsInChunk < 0 ||
        iterator->chunk->recordsInChunk > MAX_RECORDS) {
        return -1; // CWE-787
    }

    if (iterator->currentRecord >= iterator->chunk->recordsInChunk) {
        return -1; // No more records
    }

    int idx = iterator->currentRecord;

    // Safe bounded copies (CWE-120/121/787)
    bounded_copy(record->name, sizeof(record->name), iterator->chunk->records[idx].name);
    bounded_copy(record->surname, sizeof(record->surname), iterator->chunk->records[idx].surname);
    bounded_copy(record->city, sizeof(record->city), iterator->chunk->records[idx].city);
    bounded_copy(record->delimiter, sizeof(record->delimiter), iterator->chunk->records[idx].delimiter);
    record->id = iterator->chunk->records[idx].id;

    iterator->currentRecord++;
    return 0;
}

int main(void) {
    int file_desc;
    const char *filename = "example_file.txt";

    // Check return values (CWE-252)
    if (HP_CreateFile(filename) != 0) {
        fprintf(stderr, "HP_CreateFile failed for %s\n", filename);
        return EXIT_FAILURE;
    }
    if (HP_OpenFile(filename, &file_desc) != 0) {
        fprintf(stderr, "HP_OpenFile failed for %s\n", filename);
        return EXIT_FAILURE;
    }

    // Iterate over chunks
    CHUNK_Iterator chunkIterator = CHUNK_CreateIterator(file_desc, 1 /* demo limit */);
    CHUNK chunk;

    while (CHUNK_GetNext(&chunkIterator, &chunk) == 0) {
        CHUNK_Print(chunk);

        // Retrieve and print records in the CHUNK
        Record record;
        CHUNK_RecordIterator recordIterator = CHUNK_CreateRecordIterator(&chunk);
        while (CHUNK_GetNextRecord(&recordIterator, &record) == 0) {
            printf("Name: %s, Surname: %s, City: %s, ID: %d\n",
                   record.name, record.surname, record.city, record.id);
        }
    }

    if (HP_CloseFile(file_desc) != 0) {
        fprintf(stderr, "HP_CloseFile failed for fd=%d\n", file_desc);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}