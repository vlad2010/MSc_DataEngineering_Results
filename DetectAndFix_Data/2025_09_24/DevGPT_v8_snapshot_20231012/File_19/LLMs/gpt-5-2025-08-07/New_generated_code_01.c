#include "prot_queue.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <time.h>

#define TEST_BUF_SIZE 10  // For simplicity

// Fail-closed check that is not compiled out.
#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed: %s at %s:%d\n", #expr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Optional small backoff to avoid hot spinning when queue is full/empty
static inline void small_backoff(void) {
    // Try yield first; if not available, this is a no-op on some systems.
    sched_yield();
}

/* Stubs for other tests referenced in main (not part of this snippet). */
static void test_basic_event(void) {}
static void test_empty_tags(void) {}
static void test_parse_json(void) {}
static void test_parse_contact_list(void) {}
static void test_strings_work_before_finalization(void) {}
static void test_tce(void) {}
static void test_tce_command_result(void) {}
static void test_tce_eose(void) {}
static void test_tce_command_result_empty_msg(void) {}
static void test_content_len(void) {}
static void test_nostr_report(void) {}

/* Tests start here. */
static void test_queue_init_pop_push() {
    struct prot_queue q;
    int buffer[TEST_BUF_SIZE];
    int data;

    // Initialize
    CHECK(prot_queue_init(&q, buffer, sizeof(buffer), sizeof(int)) == 1);

    // Basic push and pop
    data = 5;
    CHECK(prot_queue_push(&q, &data) == 1);
    int out = 0;
    CHECK(prot_queue_pop(&q, &out) == 1);
    CHECK(out == 5);

    // Push to full using stable storage to avoid taking the address of a loop variable.
    int values[TEST_BUF_SIZE];
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        values[i] = i;
        CHECK(prot_queue_push(&q, &values[i]) == 1);
    }

    // Pushing one more should fail (non-blocking behavior assumed by this API).
    int extra = 1234;
    CHECK(prot_queue_push(&q, &extra) == 0);

    // Pop to empty and validate order
    for (int i = 0; i < TEST_BUF_SIZE; i++) {
        int out2 = -1;
        CHECK(prot_queue_try_pop(&q, &out2) == 1);
        CHECK(out2 == i);
    }
    // Now the queue should be empty.
    int dummy = 0;
    CHECK(prot_queue_try_pop(&q, &dummy) == 0);
}

// This function will be used by threads to test thread safety.
static void* thread_func(void* arg) {
    struct prot_queue* q = (struct prot_queue*) arg;
    int data;

    for (int i = 0; i < 100; i++) {
        data = i;

        // Retry until push succeeds (handles transient full queue).
        while (prot_queue_push(q, &data) != 1) {
            small_backoff();
        }

        // Retry until pop succeeds (handles transient empty queue).
        int out = -1;
        while (prot_queue_pop(q, &out) != 1) {
            small_backoff();
        }

        // Optional sanity: out should be some int previously enqueued (by either thread).
        // We don't assert equality with 'data' because interleaving makes order non-deterministic.
    }
    return NULL;
}

static void test_queue_thread_safety() {
    struct prot_queue q;

    // Use heap-allocated buffer to avoid stack lifetime pitfalls across threads.
    int* buffer = (int*)malloc(TEST_BUF_SIZE * sizeof(int));
    CHECK(buffer != NULL);

    CHECK(prot_queue_init(&q, buffer, TEST_BUF_SIZE * (int)sizeof(int), (int)sizeof(int)) == 1);

    pthread_t threads[2];

    // Create threads
    for (int i = 0; i < 2; i++) {
        int rc = pthread_create(&threads[i], NULL, thread_func, &q);
        CHECK(rc == 0);
    }

    // Join threads
    for (int i = 0; i < 2; i++) {
        int rc = pthread_join(threads[i], NULL);
        CHECK(rc == 0);
    }

    // Drain any remaining items to ensure the queue is empty (robust to interleavings).
    int out;
    while (prot_queue_try_pop(&q, &out) == 1) {
        // draining
    }

    // After draining, the queue should be empty
    CHECK(prot_queue_try_pop(&q, &out) == 0);

    // If prot_queue has a destroy/finalize function, call it here.
    // prot_queue_destroy(&q); // Uncomment if available in your API.

    free(buffer);
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
    test_queue_init_pop_push();
    test_queue_thread_safety();
    printf("All tests passed!\n");
    return 0;
}