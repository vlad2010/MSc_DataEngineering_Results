#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

// Assume the definitions for `struct ssmap`, `struct node`, `struct way`, and MinHeap functions are provided elsewhere.
// This code assumes the following minimal fields exist (commonly present in such models):
//   m->nr_nodes, m->nodes[], m->ways[]
//   struct node: int num_ways; int *way_ids;
//   struct way: int num_nodes; int *node_ids; double speed_limit;
// If your ssmap also has m->nr_ways, enable the bound check below by defining HAS_NR_WAYS and ensuring the field exists.
#define HAS_NR_WAYS 1

// Placeholder structure declarations (not definitions) to satisfy the compiler if needed.
// Remove these if your project already provides proper definitions.
struct node {
    int num_ways;
    int *way_ids;
};
struct way {
    int num_nodes;
    int *node_ids;
    double speed_limit;
};
struct ssmap {
    int nr_nodes;
#ifdef HAS_NR_WAYS
    int nr_ways;
#endif
    struct node *nodes;
    struct way  *ways;
};

// Placeholder MinHeap types and function signatures
typedef struct MinHeap MinHeap;
typedef struct { int node_id; double priority; } HeapNode;
double calculate_travel_time(struct node node1, struct node node2, double speed_limit);
double ssmap_path_travel_time(const struct ssmap * m, int size, int node_ids[]);
void min_heap_insert(MinHeap *heap, int node_id, double priority);
HeapNode min_heap_extract_min(MinHeap *heap);
bool is_min_heap_empty(MinHeap *heap);
MinHeap* create_min_heap(int capacity);
void min_heap_decrease_key(MinHeap *heap, int node_id, double priority);
void free_min_heap(MinHeap *heap);

// Safe allocation helpers
static int check_mul_overflow_size(size_t count, size_t size) {
    if (count == 0 || size == 0) return 0;
    if (count > SIZE_MAX / size) return -1;
    return 0;
}
static void *safe_calloc(size_t count, size_t size) {
    if (check_mul_overflow_size(count, size) != 0) {
        errno = EOVERFLOW;
        return NULL;
    }
    return calloc(count, size);
}

static inline bool is_valid_index(int idx, int upper_bound) {
    return idx >= 0 && (unsigned)idx < (unsigned)upper_bound;
}

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    // Basic input validation
    if (m == NULL || m->nodes == NULL || m->ways == NULL) {
        fprintf(stderr, "ssmap_path_create: invalid map pointers\n");
        return;
    }
    if (m->nr_nodes <= 0) {
        fprintf(stderr, "ssmap_path_create: no nodes available\n");
        return;
    }
    if (!is_valid_index(start_id, m->nr_nodes) || !is_valid_index(end_id, m->nr_nodes)) {
        fprintf(stderr, "ssmap_path_create: start_id or end_id out of range\n");
        return;
    }

    const int n = m->nr_nodes;
    double *travelTimes = NULL;
    int *predecessors = NULL;
    bool *visited = NULL;
    int *currentBestPath = NULL;
    int *tempPath = NULL;
    MinHeap *heap = NULL;

    // Allocate arrays with overflow checks
    travelTimes = (double *)safe_calloc((size_t)n, sizeof(double));
    predecessors = (int *)safe_calloc((size_t)n, sizeof(int));
    visited = (bool *)safe_calloc((size_t)n, sizeof(bool));
    currentBestPath = (int *)safe_calloc((size_t)n, sizeof(int));
    tempPath = (int *)safe_calloc((size_t)n, sizeof(int));

    if (!travelTimes || !predecessors || !visited || !currentBestPath || !tempPath) {
        fprintf(stderr, "ssmap_path_create: memory allocation failed\n");
        goto cleanup;
    }

    // Initialize arrays
    for (int i = 0; i < n; i++) {
        travelTimes[i] = DBL_MAX;
        predecessors[i] = -1;
        visited[i] = false;
        currentBestPath[i] = -1;
        tempPath[i] = -1;
    }

    heap = create_min_heap(n);
    if (!heap) {
        fprintf(stderr, "ssmap_path_create: failed to create min heap\n");
        goto cleanup;
    }

    // Initialize Dijkstra frontier
    for (int i = 0; i < n; i++) {
        if (i == start_id) {
            min_heap_insert(heap, i, 0.0);
            travelTimes[i] = 0.0;
        } else {
            min_heap_insert(heap, i, DBL_MAX);
        }
    }

    // Dijkstra's algorithm
    while (!is_min_heap_empty(heap)) {
        HeapNode currentNode = min_heap_extract_min(heap);
        int currentNodeId = currentNode.node_id;

        if (!is_valid_index(currentNodeId, n)) {
            // Defensive: ignore invalid IDs from heap
            continue;
        }
        if (visited[currentNodeId]) {
            continue; // Already finalized
        }
        visited[currentNodeId] = true;

        // Early exit if reached destination
        if (currentNodeId == end_id) {
            break;
        }

        struct node *currentNodePtr = &m->nodes[currentNodeId];
        // Validate num_ways non-negative and reasonable
        if (currentNodePtr->num_ways < 0) {
            fprintf(stderr, "ssmap_path_create: node %d has negative num_ways\n", currentNodeId);
            continue;
        }

        for (int i = 0; i < currentNodePtr->num_ways; i++) {
            int wayIndex = currentNodePtr->way_ids ? currentNodePtr->way_ids[i] : -1;

            if (wayIndex < 0) {
                continue; // skip invalid negative indices
            }
#ifdef HAS_NR_WAYS
            if (!is_valid_index(wayIndex, m->nr_ways)) {
                // Invalid way index; skip to avoid OOB
                continue;
            }
#endif
            struct way *wayPtr = &m->ways[wayIndex];

            if (wayPtr->num_nodes < 0) {
                // Skip corrupt way
                continue;
            }

            for (int j = 0; j < wayPtr->num_nodes; j++) {
                int neighborNodeId = wayPtr->node_ids ? wayPtr->node_ids[j] : -1;

                if (!is_valid_index(neighborNodeId, n)) {
                    continue; // skip invalid neighbor indices
                }
                if (visited[neighborNodeId]) {
                    continue;
                }
                if (travelTimes[currentNodeId] == DBL_MAX) {
                    continue; // unreachable so far
                }

                double travelTime = calculate_travel_time(m->nodes[currentNodeId], m->nodes[neighborNodeId], wayPtr->speed_limit);
                if (!isfinite(travelTime) || travelTime < 0.0) {
                    // Defensive: ignore invalid travel times
                    continue;
                }

                double candidate = travelTimes[currentNodeId] + travelTime;
                if (candidate < travelTimes[neighborNodeId]) {
                    travelTimes[neighborNodeId] = candidate;
                    predecessors[neighborNodeId] = currentNodeId;
                    min_heap_decrease_key(heap, neighborNodeId, candidate);
                }
            }
        }
    }

    // Path reconstruction with cycle/length guard
    int tempPathSize = 0;
    if (start_id == end_id) {
        tempPath[tempPathSize++] = start_id;
    } else if (predecessors[end_id] == -1) {
        // Unreachable destination
        fprintf(stderr, "No path found from %d to %d\n", start_id, end_id);
        // Optional: print empty line or nothing depending on requirements
        goto print_and_cleanup;
    } else {
        int steps = 0;
        for (int at = end_id; at != -1; ) {
            if (!is_valid_index(at, n)) {
                fprintf(stderr, "ssmap_path_create: corrupt predecessor index %d\n", at);
                tempPathSize = 0;
                break;
            }
            if (tempPathSize >= n) {
                // Cycle or corrupted predecessor chain
                fprintf(stderr, "ssmap_path_create: path exceeds node count (cycle?)\n");
                tempPathSize = 0;
                break;
            }
            tempPath[tempPathSize++] = at;
            int next = predecessors[at];
            // Guard against infinite loops via steps as well
            if (++steps > n) {
                fprintf(stderr, "ssmap_path_create: predecessor chain too long\n");
                tempPathSize = 0;
                break;
            }
            at = next;
        }
    }

    // If we have a candidate path, compute travel time and rebuild in forward order
    double bestPathTime = DBL_MAX;
    int currentBestPathSize = 0;
    if (tempPathSize > 0) {
        // The tempPath currently holds [end ... start]. Reverse into currentBestPath.
        for (int i = 0; i < tempPathSize; i++) {
            currentBestPath[i] = tempPath[tempPathSize - 1 - i];
        }
        currentBestPathSize = tempPathSize;

        // Compute travel time for sanity/logging; guard against invalid inputs
        double tempPathTime = ssmap_path_travel_time(m, currentBestPathSize, currentBestPath);
        if (isfinite(tempPathTime) && tempPathTime >= 0.0) {
            bestPathTime = tempPathTime;
        }
    }

print_and_cleanup:
    // Print the best path (if any)
    if (currentBestPathSize > 0) {
        for (int i = 0; i < currentBestPathSize; i++) {
            printf("%d ", currentBestPath[i]);
        }
        printf("\n");
        // Optionally, print travel time:
        // printf("Travel time: %.6f\n", bestPathTime);
    } else {
        // Print nothing or a newline to maintain behavior; choose one:
        printf("\n");
    }

cleanup:
    if (heap) free_min_heap(heap);
    free(travelTimes);
    free(predecessors);
    free(visited);
    free(currentBestPath);
    free(tempPath);
}