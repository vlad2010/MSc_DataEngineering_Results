#include "hashdb.h"
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 1000

// Define your hash table structure
typedef struct {
    pthread_rwlock_t lock; // Reader-Writer lock for synchronization
    hashRecord *table[TABLE_SIZE]; // Array of pointers to linked lists
} ConcurrentHashMap;

// Global hash table instance
ConcurrentHashMap hash_table;

// Initialize the hash table
void init_hash_table() {
    for (int i = 0; i < TABLE_SIZE; i++) {
        hash_table.table[i] = NULL;
    }
    pthread_rwlock_init(&hash_table.lock, NULL);
}

// Hash function (Jenkins one-at-a-time hash)
uint32_t hash_function(const char *key) {
    uint32_t hash = 0;
    size_t i;
    for (i = 0; key[i] != '\0'; ++i) {
        hash += key[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

// Function to insert or update a record in the hash table
void insert_record(const char *name, uint32_t salary) {
    uint32_t hash_value = hash_function(name) % TABLE_SIZE;

    pthread_rwlock_wrlock(&hash_table.lock);
    // Find the position in the table
    hashRecord *current = hash_table.table[hash_value];
    hashRecord *prev = NULL;
    while (current != NULL && strcmp(current->name, name) != 0) {
        prev = current;
        current = current->next;
    }

    // If record exists, update salary
    if (current != NULL && strcmp(current->name, name) == 0) {
        current->salary = salary;
    } else { // Else, create new record
        hashRecord *new_record = (hashRecord *)malloc(sizeof(hashRecord));
        if (new_record != NULL) {
            new_record->hash = hash_value;
            strncpy(new_record->name, name, 50);
            new_record->name[49] = '\0'; // Ensure null-terminated string
            new_record->salary = salary;
            new_record->next = NULL;

            // Insert at head of linked list
            if (prev == NULL) {
                hash_table.table[hash_value] = new_record;
            } else {
                prev->next = new_record;
            }
        }
    }
    pthread_rwlock_unlock(&hash_table.lock);
}

// Function to delete a record from the hash table
void delete_record(const char *name) {
    uint32_t hash_value = hash_function(name) % TABLE_SIZE;

    pthread_rwlock_wrlock(&hash_table.lock);
    hashRecord *current = hash_table.table[hash_value];
    hashRecord *prev = NULL;

    while (current != NULL && strcmp(current->name, name) != 0) {
        prev = current;
        current = current->next;
    }

    if (current != NULL && strcmp(current->name, name) == 0) {
        if (prev == NULL) {
            hash_table.table[hash_value] = current->next;
        } else {
            prev->next = current->next;
        }
        free(current);
    }
    pthread_rwlock_unlock(&hash_table.lock);
}

// Function to search for a record in the hash table
uint32_t search_record(const char *name) {
    uint32_t hash_value = hash_function(name) % TABLE_SIZE;
    uint32_t salary = 0;

    pthread_rwlock_rdlock(&hash_table.lock);
    hashRecord *current = hash_table.table[hash_value];

    while (current != NULL && strcmp(current->name, name) != 0) {
        current = current->next;
    }

    if (current != NULL && strcmp(current->name, name) == 0) {
        salary = current->salary;
    }
    pthread_rwlock_unlock(&hash_table.lock);

    return salary;
}

// Function to print a single record
void print_record(const hashRecord *record) {
    printf("%u,%s,%u\n", record->hash, record->name, record->salary);
}

// Function to print the entire hash table
void print_table() {
    pthread_rwlock_rdlock(&hash_table.lock);
    for (int i = 0; i < TABLE_SIZE; i++) {
        hashRecord *current = hash_table.table[i];
        while (current != NULL) {
            print_record(current);
            current = current->next;
        }
    }
    pthread_rwlock_unlock(&hash_table.lock);
}