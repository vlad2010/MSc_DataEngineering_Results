strcpy(record->name, "John");
strcpy(record->surname, "Doe");
strcpy(record->city, "New York");
...
strcpy(chunk->records[i].name, record.name);
strcpy(chunk->records[i].surname, record.surname);
strcpy(chunk->records[i].city, record.city);
...
strcpy(record->name, iterator->chunk.records[iterator->cursor].name);
strcpy(record->surname, iterator->chunk.records[iterator->cursor].surname);
strcpy(record->city, iterator->chunk.records[iterator->cursor].city);