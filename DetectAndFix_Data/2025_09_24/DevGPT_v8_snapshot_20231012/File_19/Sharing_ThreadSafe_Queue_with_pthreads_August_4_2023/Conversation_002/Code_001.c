#ifndef PROT_QUEUE_H
#define PROT_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "cursor.h"

#define BUFFER_SIZE 100

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
    // Ensure buffer elements fit nicely in the buffer.
	if (buflen == 0 || buflen % elem_size != 0)
		return 0;

	q->head = 0;
	q->tail = 0;
	q->count = 0;
	q->buf = buf;
	q->buflen = buflen;
	q->elem_size = elem_size;

	pthread_mutex_init(&q->mutex, NULL);
	pthread_cond_init(&q->cond, NULL);

	return 1;
}

/* 
 * Return the capacity of the queue.
 * Params:
 * q - Pointer to the queue.
 */
static inline int prot_queue_capacity(struct prot_queue *q) {
	return q->buflen / q->elem_size;
}

/* 
 * Push an element onto the queue.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to the data element to be pushed.
 * Returns 1 if successful, 0 if the queue is full.
 */
static inline int prot_queue_push(struct prot_queue* q, void *data)
{
	int cap;

	pthread_mutex_lock(&q->mutex);

	cap = prot_queue_capacity(q);
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

/* 
 * Try to pop an element from the queue without blocking.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to where the popped data will be stored.
 * Returns 1 if successful, 0 if the queue is empty.
 */
static inline int prot_queue_try_pop(struct prot_queue *q, void *data) {
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

/* 
 * Pop an element from the queue. Blocks if the queue is empty.
 * Params:
 * q    - Pointer to the queue.
 * data - Pointer to where the popped data will be stored.
 */
static inline void prot_queue_pop(struct prot_queue *q, void *data) {
	pthread_mutex_lock(&q->mutex);

	while (q->count == 0)
		pthread_cond_wait(&q->cond, &q->mutex);

	memcpy(data, &q->buf[q->head * q->elem_size], q->elem_size);
	q->head = (q->head + 1) % prot_queue_capacity(q);
	q->count--;

	pthread_cond_signal(&q->cond);
	pthread_mutex_unlock(&q->mutex);
}

/* 
 * Destroy the queue. Releases resources associated with the queue.
 * Params:
 * q - Pointer to the queue.
 */
static inline void prot_queue_destroy(struct prot_queue* q) {
	pthread_mutex_destroy(&q->mutex);
	pthread_cond_destroy(&q->cond);
}

#endif // PROT_QUEUE_H
