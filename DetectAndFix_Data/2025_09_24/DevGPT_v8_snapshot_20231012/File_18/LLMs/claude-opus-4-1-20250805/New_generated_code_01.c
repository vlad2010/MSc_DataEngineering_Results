#ifndef PROT_QUEUE_H
#define PROT_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include "cursor.h"

#define BUFFER_SIZE 100
#define MAX_ELEM_SIZE 65536  // Maximum reasonable element size
#define MAX_BUFFER_SIZE (1024 * 1024 * 100)  // 100MB max buffer

/* 
 * The prot_queue structure represents a thread-safe queue that can hold generic data elements.
 */
struct prot_queue {
    unsigned char *buf;           // Buffer to hold the data elements.
    int buflen;                   // Length of the buffer.
    int head;                     // Index of the front of the queue.
    int tail;                     // Index of the end of the queue.
    int count;                    // Current count of elements in the queue.
    int elem_size;                // Size of each data element in the buffer.
    int capacity;                 // Pre-calculated capacity
    bool initialized;             // Flag to track initialization status
    pthread_mutex_t mutex;        // Mutex to protect concurrent access.
    pthread_cond_t cond;          // Condition variable for synchronization.
};

/* 
 * Initialize the queue. 
 * Params:
 * q         - Pointer to the queue.
 * buf       - Buffer for holding data elements.
 * buflen    - Length of the buffer.
 * elem_size - Size of each data element.
 * Returns 1 if successful, 0 otherwise.
 */
static inline int prot_queue_init(struct prot_queue* q, void* buf, int buflen,
                                  int elem_size)
{
    // Validate input parameters
    if (!q || !buf) {
        errno = EINVAL;
        return 0;
    }

    // Check for already initialized queue
    if (q->initialized) {
        errno = EALREADY;
        return 0;
    }

    // Validate buffer length and element size
    if (buflen <= 0 || elem_size <= 0) {
        errno = EINVAL;
        return 0;
    }

    // Check for reasonable limits
    if (elem_size > MAX_ELEM_SIZE || buflen > MAX_BUFFER_SIZE) {
        errno = EINVAL;
        return 0;
    }

    // Ensure buffer elements fit nicely in the buffer
    if (buflen % elem_size != 0) {
        errno = EINVAL;
        return 0;
    }

    // Calculate capacity safely
    int capacity = buflen / elem_size;
    if (capacity <= 0 || capacity > INT_MAX / elem_size) {
        errno = EOVERFLOW;
        return 0;
    }

    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->buf = buf;
    q->buflen = buflen;
    q->elem_size = elem_size;
    q->capacity = capacity;
    q->initialized = true;

    if (pthread_mutex_init(&q->mutex, NULL) != 0) {
        q->initialized = false;
        return 0;
    }

    if (pthread_cond_init(&q->cond, NULL) != 0) {
        pthread_mutex_destroy(&q->mutex);
        q->initialized = false;
        return 0;
    }

    return 1;
}

/* 
 * Return the capacity of the queue.
 * Params:
 * q - Pointer to the queue.
 */
static inline int prot_queue_capacity(struct prot_queue *q) {
    if (!q || !q->initialized) {
        return 0;
    }
    // Return pre-calculated capacity to avoid division
    return q->capacity;
}

/* 
 * Safely calculate buffer offset
 */
static inline size_t safe_buffer_offset(int index, int elem_size, int capacity) {
    // Ensure index is within bounds
    if (index < 0 || index >= capacity) {
        return 0;
    }
    // Use size_t to avoid integer overflow
    return (size_t)index * (size_t)elem_size;
}

/* 
 * Push an element onto the queue.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to the data element to be pushed.
 * Returns 1 if successful, 0 if the queue is full or error.
 */
static inline int prot_queue_push(struct prot_queue* q, void *data)
{
    if (!q || !data || !q->initialized) {
        errno = EINVAL;
        return 0;
    }

    pthread_mutex_lock(&q->mutex);

    if (q->count == q->capacity) {
        pthread_mutex_unlock(&q->mutex);
        errno = ENOBUFS;
        return 0;
    }

    size_t offset = safe_buffer_offset(q->tail, q->elem_size, q->capacity);
    if (offset + q->elem_size > (size_t)q->buflen) {
        pthread_mutex_unlock(&q->mutex);
        errno = EOVERFLOW;
        return 0;
    }

    memcpy(&q->buf[offset], data, q->elem_size);
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);

    return 1;
}

/* 
 * Try to pop an element from the queue without blocking.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to where the popped data will be stored.
 * Returns 1 if successful, 0 if the queue is empty or error.
 */
static inline int prot_queue_try_pop(struct prot_queue *q, void *data) {
    if (!q || !data || !q->initialized) {
        errno = EINVAL;
        return 0;
    }

    pthread_mutex_lock(&q->mutex);

    if (q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        errno = EAGAIN;
        return 0;
    }

    size_t offset = safe_buffer_offset(q->head, q->elem_size, q->capacity);
    if (offset + q->elem_size > (size_t)q->buflen) {
        pthread_mutex_unlock(&q->mutex);
        errno = EOVERFLOW;
        return 0;
    }

    memcpy(data, &q->buf[offset], q->elem_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

/* 
 * Pop an element from the queue. Blocks if the queue is empty.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to where the popped data will be stored.
 * Returns 1 on success, 0 on error
 */
static inline int prot_queue_pop(struct prot_queue *q, void *data) {
    if (!q || !data || !q->initialized) {
        errno = EINVAL;
        return 0;
    }

    pthread_mutex_lock(&q->mutex);

    while (q->count == 0) {
        if (pthread_cond_wait(&q->cond, &q->mutex) != 0) {
            pthread_mutex_unlock(&q->mutex);
            return 0;
        }
    }

    size_t offset = safe_buffer_offset(q->head, q->elem_size, q->capacity);
    if (offset + q->elem_size > (size_t)q->buflen) {
        pthread_mutex_unlock(&q->mutex);
        errno = EOVERFLOW;
        return 0;
    }

    memcpy(data, &q->buf[offset], q->elem_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

/* 
 * Destroy the queue. Releases resources associated with the queue.
 * Params:
 * q - Pointer to the queue.
 * Returns 1 on success, 0 on error
 */
static inline int prot_queue_destroy(struct prot_queue* q) {
    if (!q || !q->initialized) {
        errno = EINVAL;
        return 0;
    }

    pthread_mutex_lock(&q->mutex);
    q->initialized = false;
    pthread_mutex_unlock(&q->mutex);

    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);
    
    // Clear sensitive data
    if (q->buf) {
        memset(q->buf, 0, q->buflen);
    }
    memset(q, 0, sizeof(struct prot_queue));
    
    return 1;
}

#endif // PROT_QUEUE_H