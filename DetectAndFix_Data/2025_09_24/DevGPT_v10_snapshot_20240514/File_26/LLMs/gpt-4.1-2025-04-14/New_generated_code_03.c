for (i = 0; i < chunk.recordsInChunk; i++) {
    printf("Record %d: %s %s\n", i + 1, chunk.records[i].name, chunk.records[i].surname);
}