// Creating insert function where we insert or update the data for the given name and salary value.
void insert(const char *name, int salary) {
    // Lock the hash table for writing
    pthread_rwlock_wrlock(&hash_table.lock);
    // Search for the record
    hashRecord *current = hash_table.head;
    hashRecord *prev = NULL;
    while (current != NULL) {
        if (strcmp(current->name, name) == 0) {
            // Update the salary
            current->salary = salary;
            // Unlock the hash table
            pthread_rwlock_unlock(&hash_table.lock);
            return;
        }
        prev = current;
        current = current->next;
    }
    // Create a new record
    hashRecord *new_record = (hashRecord *)malloc(sizeof(hashRecord));
    strncpy(new_record->name, name, 50);
    new_record->name[49] = '\0'; // Ensure null-terminated string
    new_record->salary = salary;
    new_record->next = NULL;
    // Insert the new record
    if (prev == NULL) {
        hash_table.head = new_record;
    } else {
        prev->next = new_record;
    }
    // Unlock the hash table
    pthread_rwlock_unlock(&hash_table.lock);
}