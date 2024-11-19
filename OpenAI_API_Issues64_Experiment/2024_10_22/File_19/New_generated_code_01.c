#include "prot_queue.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_BUF_SIZE 10  // For simplicity

// Mutex for synchronizing access to the queue
pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;

// Error handling function
void handle_error(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static void test_queue_init_pop_push() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    int data;

    // Initialize
    if (prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) != 1) {
        handle_error("Failed to initialize queue");
    }

    // Push and Pop
    data = 5;
    if (prot_queue_push(&q, &data) != 1) {
        handle_error("Failed to push data");
    }
    if (prot_queue_pop(&q, &data) != 1 || data != 5) {
        handle_error("Failed to pop data or data mismatch");
    }

    // Push to full, and then fail to push
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        if (prot_queue_push(&q, &i) != 1) {
            handle_error("Queue push failed prematurely");
        }
    }
    if (prot_queue_push(&q, &data) != 0) {  // Should fail as queue is full
        handle_error("Queue push succeeded even when full");
    }

    // Pop to empty, and then fail to pop
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        if (prot_queue_try_pop(&q, &data) != 1 || data != i) {
            handle_error("Failed to pop data correctly or data mismatch");
        }
    }
    if (prot_queue_try_pop(&q, &data) != 0) {  // Should fail as queue is empty
        handle_error("Queue pop succeeded even when empty");
    }
}

void* thread_func(void* arg) {
    struct prot_queue* q = (struct prot_queue*) arg;
    int data;

    for (int i = 0; i < 100; i++) {
        data = i;

        pthread_mutex_lock(&queue_mutex);
        prot_queue_push(q, &data);
        prot_queue_pop(q, &data);
        pthread_mutex_unlock(&queue_mutex);
    }
    return NULL;
}

static void test_queue_thread_safety() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    pthread_t threads[2];

    if (prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) != 1) {
        handle_error("Failed to initialize queue");
    }

    // Create threads
    for (int i = 0; i < 2; i++) {
        if (pthread_create(&threads[i], NULL, thread_func, &q) != 0) {
            handle_error("Failed to create thread");
        }
    }

    // Join threads
    for (int i = 0; i < 2; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            handle_error("Failed to join thread");
        }
    }

    // After all operations, the queue should be empty
    int data;
    pthread_mutex_lock(&queue_mutex);
    if (prot_queue_try_pop(&q, &data) != 0) {
        handle_error("Queue is not empty after operations");
    }
    pthread_mutex_unlock(&queue_mutex);
}

int main(int argc, const char *argv[]) {
    // Presumed function declarations; these should be defined elsewhere.
    extern void test_basic_event(void);
    extern void test_empty_tags(void);
    extern void test_parse_json(void);
    extern void test_parse_contact_list(void);
    extern void test_strings_work_before_finalization(void);
    extern void test_tce(void);
    extern void test_tce_command_result(void);
    extern void test_tce_eose(void);
    extern void test_tce_command_result_empty_msg(void);
    extern void test_content_len(void);
    extern void test_nostr_report(void);

    test_basic_event();
    test_empty_tags();
    test_parse_json();
    test_parse_contact_list();
    test_strings_work_before_finalization();
    test_tce();
    test_tce_command_result();
    test_tce_eose();
    test_tce_command_result_empty_msg();
    test_content_len();
    test_nostr_report();
    test_queue_init_pop_push();          // Added
    test_queue_thread_safety();          // Added
    printf("All tests passed!\n");       // Print this if all tests pass.

    pthread_mutex_destroy(&queue_mutex); // Cleanup the mutex resource
}