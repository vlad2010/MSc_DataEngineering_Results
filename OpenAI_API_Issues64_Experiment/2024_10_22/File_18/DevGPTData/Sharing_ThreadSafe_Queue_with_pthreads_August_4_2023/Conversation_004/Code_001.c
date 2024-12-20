#include "prot_queue.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_BUF_SIZE 10  // For simplicity

static void test_queue_init_pop_push() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    int data;

    // Initialize
    assert(prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) == 1);

    // Push and Pop
    data = 5;
    assert(prot_queue_push(&q, &data) == 1);
    assert(prot_queue_pop(&q, &data) == 1);
    assert(data == 5);

    // Push to full, and then fail to push
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        assert(prot_queue_push(&q, &i) == 1);
    }
    assert(prot_queue_push(&q, &data) == 0);  // Should fail as queue is full

    // Pop to empty, and then fail to pop
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        assert(prot_queue_try_pop(&q, &data) == 1);
        assert(data == i);
    }
    assert(prot_queue_try_pop(&q, &data) == 0);  // Should fail as queue is empty
}

// This function will be used by threads to test thread safety.
void* thread_func(void* arg) {
    struct prot_queue* q = (struct prot_queue*) arg;
    int data;

    for (int i = 0; i < 100; i++) {
        data = i;
        prot_queue_push(q, &data);
        prot_queue_pop(q, &data);
    }
    return NULL;
}

static void test_queue_thread_safety() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    pthread_t threads[2];

    assert(prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) == 1);

    // Create threads
    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, thread_func, &q);
    }

    // Join threads
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }

    // After all operations, the queue should be empty
    int data;
    assert(prot_queue_try_pop(&q, &data) == 0);
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
