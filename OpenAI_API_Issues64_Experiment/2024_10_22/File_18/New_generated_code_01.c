#ifndef PROT_QUEUE_H
#define PROT_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h> // For free()
#include "cursor.h"

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

static inline int prot_queue_init(struct prot_queue* q, unsigned char* buf, int buflen, int elem_size)
{
    if (!q || !buf || buflen <= 0 || elem_size <= 0 || buflen % elem_size != 0) {
        return 0;
    }

    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->buf = buf;
    q->buflen = buflen;
    q->elem_size = elem_size;

    if (pthread_mutex_init(&q->mutex, NULL) != 0) {
        return 0;
    }
    if (pthread_cond_init(&q->cond, NULL) != 0) {
        pthread_mutex_destroy(&q->mutex);
        return 0;
    }
    return 1;
}

static inline int prot_queue_capacity(struct prot_queue *q) {
    if (!q || q->elem_size == 0) {
        return 0;
    }
    return q->buflen / q->elem_size;
}

static inline int prot_queue_push(struct prot_queue* q, void *data)
{
    if (!q || !data) {
        return 0;
    }

    pthread_mutex_lock(&q->mutex);

    int cap = prot_queue_capacity(q);
    if (q->count == cap) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    memcpy(&q->buf[q->tail * q->elem_size], data, q->elem_size);
    q->tail = (q->tail + 1) % cap;
    q->count++;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);

    return 1;
}

static inline int prot_queue_try_pop(struct prot_queue *q, void *data) {
    if (!q || !data) {
        return 0;
    }

    pthread_mutex_lock(&q->mutex);

    if (q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }

    memcpy(data, &q->buf[q->head * q->elem_size], q->elem_size);
    q->head = (q->head + 1) % prot_queue_capacity(q);
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

static inline void prot_queue_pop(struct prot_queue *q, void *data) {
    if (!q || !data) {
        return;
    }

    pthread_mutex_lock(&q->mutex);

    while (q->count == 0) {
        pthread_cond_wait(&q->cond, &q->mutex);
    }

    memcpy(data, &q->buf[q->head * q->elem_size], q->elem_size);
    q->head = (q->head + 1) % prot_queue_capacity(q);
    q->count--;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

static inline void prot_queue_destroy(struct prot_queue* q) {
    if (!q) {
        return;
    }

    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);

    if (q->buf) {
        free(q->buf);
        q->buf = NULL;
    }
}

#endif // PROT_QUEUE_H