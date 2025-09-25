#include <stdio.h>

// Assume the definitions from the previous code are included here.

int CHUNK_GetNext(CHUNK_Iterator *iterator, CHUNK *chunk);

int main() {
    // Example usage:
    int file_desc;
    char *filename = "example_file.txt";
    HP_CreateFile(filename);
    HP_OpenFile(filename, &file_desc);

    CHUNK_Iterator iterator = CHUNK_CreateIterator(file_desc, 10);
    CHUNK chunk;

    // Retrieve the next CHUNK in the sequence
    while (CHUNK_GetNext(&iterator, &chunk) == 0) {
        // Process the retrieved CHUNK
        printf("File Descriptor: %d\n", chunk.file_desc);
        printf("From Block ID: %d\n", chunk.from_BlockId);
        printf("To Block ID: %d\n", chunk.to_BlockId);
        printf("Records in Chunk: %d\n", chunk.recordsInChunk);
        printf("Blocks in Chunk: %d\n", chunk.blocksInChunk);
        printf("\n");
    }

    HP_CloseFile(file_desc);

    return 0;
}

int CHUNK_GetNext(CHUNK_Iterator *iterator, CHUNK *chunk) {
    // Check if the current block exceeds the specified range
    if (iterator->current > iterator->lastBlocksID) {
        return -1; // No more CHUNKs to retrieve
    }

    // Set CHUNK parameters based on iterator values
    chunk->file_desc = iterator->file_desc;
    chunk->from_BlockId = iterator->current;
    chunk->to_BlockId = iterator->current + iterator->blocksInChunk - 1;
    chunk->recordsInChunk = 0; // You may need to update this based on your requirements
    chunk->blocksInChunk = iterator->blocksInChunk;

    // Move to the next CHUNK
    iterator->current += iterator->blocksInChunk;

    return 0; // Success
}