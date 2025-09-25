#include "prot_queue.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_BUF_SIZE 10  // For simplicity

// Add function prototypes for all test functions
void test_basic_event(void);
void test_empty_tags(void);
void test_parse_json(void);
void test_parse_contact_list(void);
void test_strings_work_before_finalization(void);
void test_tce(void);
void test_tce_command_result(void);
void test_tce_eose(void);
void test_tce_command_result_empty_msg(void);
void test_content_len(void);
void test_nostr_report(void);

static void test_queue_init_pop_push() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    int data;

    // Initialize
    if (prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) != 1) {
        fprintf(stderr, "prot_queue_init failed\n");
        exit(EXIT_FAILURE);
    }

    // Push and Pop
    data = 5;
    if (prot_queue_push(&q, &data) != 1) {
        fprintf(stderr, "prot_queue_push failed\n");
        exit(EXIT_FAILURE);
    }
    if (prot_queue_pop(&q, &data) != 1) {
        fprintf(stderr, "prot_queue_pop failed\n");
        exit(EXIT_FAILURE);
    }
    if (data != 5) {
        fprintf(stderr, "Data mismatch after pop\n");
        exit(EXIT_FAILURE);
    }

    // Push to full, and then fail to push
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        if (prot_queue_push(&q, &i) != 1) {
            fprintf(stderr, "prot_queue_push failed at i=%d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    if (prot_queue_push(&q, &data) != 0) {  // Should fail as queue is full
        fprintf(stderr, "prot_queue_push should have failed (queue full)\n");
        exit(EXIT_FAILURE);
    }

    // Pop to empty, and then fail to pop
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        if (prot_queue_try_pop(&q, &data) != 1) {
            fprintf(stderr, "prot_queue_try_pop failed at i=%d\n", i);
            exit(EXIT_FAILURE);
        }
        if (data != i) {
            fprintf(stderr, "Data mismatch at i=%d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    if (prot_queue_try_pop(&q, &data) != 0) {  // Should fail as queue is empty
        fprintf(stderr, "prot_queue_try_pop should have failed (queue empty)\n");
        exit(EXIT_FAILURE);
    }
}

// This function will be used by threads to test thread safety.
void* thread_func(void* arg) {
    struct prot_queue* q = (struct prot_queue*) arg;
    int data;

    for (int i = 0; i < 100; i++) {
        data = i;
        if (prot_queue_push(q, &data) != 1) {
            fprintf(stderr, "prot_queue_push failed in thread\n");
            // Optionally exit or handle error
        }
        if (prot_queue_pop(q, &data) != 1) {
            fprintf(stderr, "prot_queue_pop failed in thread\n");
            // Optionally exit or handle error
        }
    }
    return NULL;
}

static void test_queue_thread_safety() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    pthread_t threads[2];

    if (prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) != 1) {
        fprintf(stderr, "prot_queue_init failed\n");
        exit(EXIT_FAILURE);
    }

    // Create threads
    for (int i = 0; i < 2; i++) {
        int rc = pthread_create(&threads[i], NULL, thread_func, &q);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed: %d\n", rc);
            exit(EXIT_FAILURE);
        }
    }

    // Join threads
    for (int i = 0; i < 2; i++) {
        int rc = pthread_join(threads[i], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join failed: %d\n", rc);
            exit(EXIT_FAILURE);
        }
    }

    // After all operations, the queue should be empty
    int data = 0;
    if (prot_queue_try_pop(&q, &data) != 0) {
        fprintf(stderr, "Queue not empty after threads\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, const char *argv[]) {
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
}