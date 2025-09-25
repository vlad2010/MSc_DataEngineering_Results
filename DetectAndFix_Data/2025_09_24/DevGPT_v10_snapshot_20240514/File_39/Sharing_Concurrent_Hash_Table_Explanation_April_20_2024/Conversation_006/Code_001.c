#include "hashdb.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h> // Include stdint.h for uint32_t

// Define your hash table structure
typedef struct {
    pthread_t lock; // Reader-Writer lock for synchronization
    hashRecord *head; // Head of the linked list
} ConcurrentHashMap;

// Global hash table instance
ConcurrentHashMap hash_table;

// Initialize the hash table
void init_hash_table() {
    hash_table.head = NULL;
    pthread_mutex_init(&hash_table.lock, NULL);
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
    uint32_t hash_value = hash_function(name);

    pthread_mutex_lock(&hash_table.lock);

    // Create new record
    hashRecord *new_record = (hashRecord *)malloc(sizeof(hashRecord));
    if (new_record != NULL) {
        new_record->hash = hash_value;
        strncpy(new_record->name, name, 50);
        new_record->name[49] = '\0'; // Ensure null-terminated string
        new_record->salary = salary;
        new_record->next = NULL;

        // Insert at head of linked list
        new_record->next = hash_table.head;
        hash_table.head = new_record;
    }

    pthread_mutex_unlock(&hash_table.lock);
}

// Function to delete a record from the hash table
void delete_record(const char *name) {
    pthread_mutex_lock(&hash_table.lock);

    hashRecord *current = hash_table.head;
    hashRecord *prev = NULL;

    while (current != NULL && strcmp(current->name, name) != 0) {
        prev = current;
        current = current->next;
    }

    if (current != NULL && strcmp(current->name, name) == 0) {
        if (prev == NULL) {
            hash_table.head = current->next;
        } else {
            prev->next = current->next;
        }
        free(current);
    }

    pthread_mutex_unlock(&hash_table.lock);
}

// Function to search for a record in the hash table
uint32_t search_record(const char *name) {
    pthread_mutex_lock(&hash_table.lock);

    hashRecord *current = hash_table.head;

    while (current != NULL && strcmp(current->name, name) != 0) {
        current = current->next;
    }

    uint32_t salary = 0;
    if (current != NULL && strcmp(current->name, name) == 0) {
        salary = current->salary;
    }

    pthread_mutex_unlock(&hash_table.lock);

    return salary;
}

// Function to print a single record
void print_record(const hashRecord *record) {
    printf("%u,%s,%u\n", record->hash, record->name, record->salary);
}

// Function to print the entire hash table
void print_table() {
    pthread_mutex_lock(&hash_table.lock);

    hashRecord *current = hash_table.head;

    while (current != NULL) {
        print_record(current);
        current = current->next;
    }

    pthread_mutex_unlock(&hash_table.lock);
}