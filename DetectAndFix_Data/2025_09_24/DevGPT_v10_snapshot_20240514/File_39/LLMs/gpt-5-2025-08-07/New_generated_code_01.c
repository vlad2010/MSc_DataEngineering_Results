#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Example record definition (since the original header is not provided)
typedef struct hashRecord {
    char name[50];           // fixed-size buffer; consider dynamic allocation if unbounded names are needed
    int salary;
    struct hashRecord *next;
} hashRecord;

// Concurrent hash "map" (actually a linked list in this snippet)
typedef struct {
    pthread_rwlock_t lock;   // FIX: correct lock type for rwlock
    hashRecord *head;
} ConcurrentHashMap;

// Global instance
static ConcurrentHashMap hash_table;

// Jenkins one-at-a-time hash (kept as-is; note: non-cryptographic)
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

// Initialize the hash table (synchronous, with error checking)
int init_hash_table(void) {
    memset(&hash_table, 0, sizeof(hash_table)); // head = NULL
    int rc = pthread_rwlock_init(&hash_table.lock, NULL);
    if (rc != 0) {
        // Handle error appropriately for your application; here we print and return error.
        fprintf(stderr, "pthread_rwlock_init failed: %s\n", strerror(rc));
        return -1;
    }
    return 0;
}

// Destroy the hash table, freeing resources
void destroy_hash_table(void) {
    // Acquire write lock to safely tear down the list
    int rc = pthread_rwlock_wrlock(&hash_table.lock);
    if (rc == 0) {
        hashRecord *cur = hash_table.head;
        while (cur) {
            hashRecord *next = cur->next;
            // Zero sensitive data if needed (not strictly necessary for salary/name)
            // explicit_bzero(cur, sizeof(*cur)); // if available and desired
            free(cur);
            cur = next;
        }
        hash_table.head = NULL;
        pthread_rwlock_unlock(&hash_table.lock);
    } else {
        // If we cannot lock, warn and continue to destroy the lock to avoid leaks
        fprintf(stderr, "pthread_rwlock_wrlock failed during destroy: %s\n", strerror(rc));
    }
    rc = pthread_rwlock_destroy(&hash_table.lock);
    if (rc != 0) {
        fprintf(stderr, "pthread_rwlock_destroy failed: %s\n", strerror(rc));
    }
}

// Insert or update a record by name
// NOTE: This keeps the original signature; errors are logged, and the function returns silently on failure.
void insert(const char *name, int salary) {
    if (name == NULL) {
        fprintf(stderr, "insert: name is NULL\n");
        return;
    }

    // Acquire write lock
    int rc = pthread_rwlock_wrlock(&hash_table.lock);
    if (rc != 0) {
        fprintf(stderr, "insert: pthread_rwlock_wrlock failed: %s\n", strerror(rc));
        return;
    }

    // Search for existing record
    hashRecord *current = hash_table.head;
    hashRecord *prev = NULL;
    while (current != NULL) {
        if (strcmp(current->name, name) == 0) {
            // Update existing
            current->salary = salary;
            pthread_rwlock_unlock(&hash_table.lock);
            return;
        }
        prev = current;
        current = current->next;
    }

    // Create a new record
    hashRecord *new_record = (hashRecord *)calloc(1, sizeof(hashRecord));
    if (new_record == NULL) {
        fprintf(stderr, "insert: allocation failed\n");
        pthread_rwlock_unlock(&hash_table.lock);
        return;
    }

    // Copy name safely. Option A: allow truncation; Option B: reject too-long names.
    // Here we choose to reject too-long names to avoid truncation ambiguity.
    size_t maxlen = sizeof(new_record->name) - 1;
    size_t inlen = strnlen(name, sizeof(new_record->name));
    if (inlen > maxlen) {
        fprintf(stderr, "insert: name too long (max %zu)\n", maxlen);
        free(new_record);
        pthread_rwlock_unlock(&hash_table.lock);
        return;
    }
    // Safe copy and NUL-terminate
    // Using snprintf ensures null-termination; inlen <= maxlen guarantees no truncation
    int n = snprintf(new_record->name, sizeof(new_record->name), "%s", name);
    if (n < 0 || (size_t)n > maxlen) {
        fprintf(stderr, "insert: name copy failed\n");
        free(new_record);
        pthread_rwlock_unlock(&hash_table.lock);
        return;
    }

    new_record->salary = salary;
    new_record->next = NULL;

    // Insert at tail or as head
    if (prev == NULL) {
        hash_table.head = new_record;
    } else {
        prev->next = new_record;
    }

    pthread_rwlock_unlock(&hash_table.lock);
}

// Example usage pattern
// int main(void) {
//     if (init_hash_table() != 0) {
//         return EXIT_FAILURE;
//     }
//     insert("alice", 100000);
//     insert("bob", 120000);
//     insert("alice", 130000); // update
//     destroy_hash_table();
//     return EXIT_SUCCESS;
// }