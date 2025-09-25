#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX_RECORDS_PER_CHUNK 1024

// Define the Record structure
struct Record {
    int  id;
    char name[15];
    char surname[25];
    char address[50];
};

// Define the CHUNK structure
struct CHUNK {
    int      file_desc;
    int      from_BlockId;
    int      to_BlockId;
    size_t   recordsInChunk;    // use size_t for counts to avoid negatives
    size_t   blocksInChunk;
    struct Record records[MAX_RECORDS_PER_CHUNK];
};

// Safe copying that guarantees null-termination and truncation
static void safe_copy(char *dst, size_t dst_sz, const char *src) {
    if (!dst || dst_sz == 0) return;
    if (!src) src = "";
    // snprintf guarantees null-termination up to dst_sz bytes
    (void)snprintf(dst, dst_sz, "%s", src);
}

// Comparator for Records: ascending by name, then surname, then id
static int cmp_records(const struct Record *a, const struct Record *b) {
    int c;
    if (a == NULL && b == NULL) return 0;
    if (a == NULL) return 1;  // put NULLs last
    if (b == NULL) return -1;

    c = strcmp(a->name, b->name);
    if (c != 0) return c;

    c = strcmp(a->surname, b->surname);
    if (c != 0) return c;

    // Optional tiebreaker to keep sort stable/deterministic
    if (a->id < b->id) return -1;
    if (a->id > b->id) return 1;
    return 0;
}

// Function to perform bubble sort on records within a chunk (ascending)
static int sort_Chunk(struct CHUNK *chunk) {
    if (chunk == NULL) return -1; // CWE-476 mitigated

    // Validate bounds to prevent OOB (CWE-787/CWE-125)
    if (chunk->recordsInChunk > MAX_RECORDS_PER_CHUNK) {
        // Reject or clamp; here we reject
        return -2;
    }

    if (chunk->recordsInChunk < 2) return 0; // nothing to do

    // Bubble sort (you may prefer qsort with cmp_records for simplicity)
    for (size_t i = 0; i < chunk->recordsInChunk - 1; i++) {
        bool swapped = false;
        for (size_t j = 0; j < chunk->recordsInChunk - i - 1; j++) {
            if (cmp_records(&chunk->records[j], &chunk->records[j + 1]) > 0) {
                struct Record tmp = chunk->records[j];
                chunk->records[j] = chunk->records[j + 1];
                chunk->records[j + 1] = tmp;
                swapped = true;
            }
        }
        if (!swapped) break; // optimization: array already sorted
    }
    return 0;
}

// Helper to initialize a record safely
static void init_record(struct Record *r, int id,
                        const char *name, const char *surname, const char *address) {
    if (!r) return;
    r->id = id;
    safe_copy(r->name, sizeof(r->name), name);
    safe_copy(r->surname, sizeof(r->surname), surname);
    safe_copy(r->address, sizeof(r->address), address);
}

int main(void) {
    // Initialize CHUNK to zeros to avoid CWE-457/CWE-665
    struct CHUNK chunk = {0};

    // Populate with sample data safely
    chunk.recordsInChunk = 5; // must not exceed MAX_RECORDS_PER_CHUNK
    init_record(&chunk.records[0], 3, "Alice", "Zephyr", "123 Main St");
    init_record(&chunk.records[1], 1, "Alice", "Anders", "9 1st Ave");
    init_record(&chunk.records[2], 2, "Bob",   "Carter", "77 Elm St");
    init_record(&chunk.records[3], 4, "Bob",   "Anders", "12 Oak St");
    init_record(&chunk.records[4], 5, "Aaron", "Young",  "3 Pine St");

    // Sort records, handle errors
    int rc = sort_Chunk(&chunk);
    if (rc != 0) {
        fprintf(stderr, "sort_Chunk failed (rc=%d)\n", rc);
        return EXIT_FAILURE;
    }

    // Display the sorted records
    for (size_t i = 0; i < chunk.recordsInChunk; i++) {
        printf("Record %zu: id=%d, %s %s, %s\n",
               i + 1,
               chunk.records[i].id,
               chunk.records[i].name,
               chunk.records[i].surname,
               chunk.records[i].address);
    }

    return EXIT_SUCCESS;
}