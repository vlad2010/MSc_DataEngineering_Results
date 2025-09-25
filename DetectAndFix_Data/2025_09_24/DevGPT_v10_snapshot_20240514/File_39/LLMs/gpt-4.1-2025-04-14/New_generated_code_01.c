#include "hashdb.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Define your hash table structure
typedef struct {
    pthread_rwlock_t lock; // Reader-Writer lock for synchronization
    hashRecord *head; // Head of the linked list
} ConcurrentHashMap;

// Global hash table instance
ConcurrentHashMap hash_table;

// Jenkins Function
uint32_t jenkins_one_at_a_time_hash(const uint8_t* key, size_t length) {
    size_t i = 0;
    uint32_t hash = 0;
    while (i != length) {
        hash += key[i++];
        hash += hash << 10;
        hash ^= hash >> 6;
    }
    hash += hash << 3;
    hash ^= hash >> 11;
    hash += hash << 15;
    return hash;
}

// Initialize the hash table and the lock
void init_hash_table() {
    hash_table.head = NULL;
    if (pthread_rwlock_init(&hash_table.lock, NULL) != 0) {
        fprintf(stderr, "Failed to initialize rwlock\n");
        exit(EXIT_FAILURE);
    }
}

// Creating insert function where we insert or update the data for the given name and salary value.
void insert(const char *name, int salary) {
    // Lock the hash table for writing
    pthread_rwlock_wrlock(&hash_table.lock);
    // Search for the record
    hashRecord *current = hash_table.head;
    hashRecord *prev = NULL;
    while (current != NULL) {
        if (strncmp(current->name, name, 50) == 0) {
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
    if (new_record == NULL) {
        // Handle allocation failure
        pthread_rwlock_unlock(&hash_table.lock);
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Safe copy: ensure null-termination
    strncpy(new_record->name, name, 49);
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

// Optional: function to free all hash records (to avoid memory leaks)
void free_hash_table() {
    pthread_rwlock_wrlock(&hash_table.lock);
    hashRecord *current = hash_table.head;
    while (current != NULL) {
        hashRecord *tmp = current;
        current = current->next;
        free(tmp);
    }
    hash_table.head = NULL;
    pthread_rwlock_unlock(&hash_table.lock);
    pthread_rwlock_destroy(&hash_table.lock);
}