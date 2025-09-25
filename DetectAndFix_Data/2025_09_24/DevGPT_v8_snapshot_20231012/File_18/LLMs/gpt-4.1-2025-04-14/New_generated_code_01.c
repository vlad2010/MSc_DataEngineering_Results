#ifndef PROT_QUEUE_H
#define PROT_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>   // For SIZE_MAX
#include <stdlib.h>   // For memset

#define BUFFER_SIZE 100

struct prot_queue {
    unsigned char *buf;
    int buflen;
    int head;
    int tail;
    int count;
    int elem_size;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

/*
 * Initialize the queue.
 * Returns 1 if successful, 0 otherwise.
 */
static inline int prot_queue_init(struct prot_queue* q, void* buf, int buflen, int elem_size)
{
    // Validate input parameters
    if (!q || !buf || buflen <= 0 || elem_size <= 0)
        return 0;

    // Check for integer overflow in division
    if (buflen % elem_size != 0)
        return 0;

    int cap = buflen / elem_size;
    if (cap <= 0)
        return 0;

    // Defensive: check for multiplication overflow
    if ((size_t)cap > SIZE_MAX / (size_t)elem_size)
        return 0;

    // Initialize structure
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->buf = buf;
    q->buflen = buflen;
    q->elem_size = elem_size;

    // Check pthread init return values
    if (pthread_mutex_init(&q->mutex, NULL) != 0) {
        memset(q, 0, sizeof(*q));
        return 0;
    }
    if (pthread_cond_init(&q->cond, NULL) != 0) {
        pthread_mutex_destroy(&q->mutex);
        memset(q, 0, sizeof(*q));
        return 0;
    }

    return 1;
}

static inline int prot_queue_capacity(struct prot_queue *q) {
    if (!q || q->elem_size <= 0) return 0;
    return q->buflen / q->elem_size;
}

static inline int prot_queue_push(struct prot_queue* q, void *data)
{
    int cap, ret = 0;
    if (!q || !data) return 0;

    if (pthread_mutex_lock(&q->mutex) != 0)
        return 0;

    cap = prot_queue_capacity(q);
    if (q->count < 0 || q->count > cap) { // Defensive
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    if (q->count == cap) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    // Defensive: check index calculation
    if (q->tail < 0 || q->tail >= cap) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    memcpy(&q->buf[(size_t)q->tail * (size_t)q->elem_size], data, (size_t)q->elem_size);
    q->tail = (q->tail + 1) % cap;
    q->count++;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);

    return 1;
}

static inline int prot_queue_try_pop(struct prot_queue *q, void *data) {
    int cap;
    if (!q || !data) return 0;

    if (pthread_mutex_lock(&q->mutex) != 0)
        return 0;

    cap = prot_queue_capacity(q);
    if (q->count <= 0) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    if (q->head < 0 || q->head >= cap) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    memcpy(data, &q->buf[(size_t)q->head * (size_t)q->elem_size], (size_t)q->elem_size);
    q->head = (q->head + 1) % cap;
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

static inline void prot_queue_pop(struct prot_queue *q, void *data) {
    int cap;
    if (!q || !data) return;

    if (pthread_mutex_lock(&q->mutex) != 0)
        return;

    cap = prot_queue_capacity(q);
    while (q->count == 0)
        pthread_cond_wait(&q->cond, &q->mutex);

    if (q->head < 0 || q->head >= cap) {
        pthread_mutex_unlock(&q->mutex);
        return;
    }

    memcpy(data, &q->buf[(size_t)q->head * (size_t)q->elem_size], (size_t)q->elem_size);
    q->head = (q->head + 1) % cap;
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

static inline void prot_queue_destroy(struct prot_queue* q) {
    if (!q) return;
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);
}

#endif // PROT_QUEUE_H