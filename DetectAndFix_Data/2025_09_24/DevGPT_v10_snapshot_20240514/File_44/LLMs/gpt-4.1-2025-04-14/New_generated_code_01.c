#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>

// Assume the definitions for `struct ssmap`, `struct node`, `struct way`, and MinHeap functions are provided elsewhere.

// Placeholder function signatures
double calculate_travel_time(struct node node1, struct node node2, double speed_limit);
double ssmap_path_travel_time(const struct ssmap * m, int size, int node_ids[]);
void min_heap_insert(MinHeap *heap, int node_id, double priority);
HeapNode min_heap_extract_min(MinHeap *heap);
bool is_min_heap_empty(MinHeap *heap);
MinHeap* create_min_heap(int capacity);
void min_heap_decrease_key(MinHeap *heap, int node_id, double priority);
void free_min_heap(MinHeap *heap); // Assume this is implemented

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    // Validate input
    if (!m || m->nr_nodes <= 0 || start_id < 0 || end_id < 0 ||
        start_id >= m->nr_nodes || end_id >= m->nr_nodes) {
        fprintf(stderr, "Invalid input parameters.\n");
        return;
    }

    // Prevent integer overflow in malloc
    if (m->nr_nodes > INT_MAX / (int)sizeof(double) ||
        m->nr_nodes > INT_MAX / (int)sizeof(int) ||
        m->nr_nodes > INT_MAX / (int)sizeof(bool)) {
        fprintf(stderr, "Too many nodes.\n");
        return;
    }

    double *travelTimes = malloc(m->nr_nodes * sizeof(double));
    int *predecessors = malloc(m->nr_nodes * sizeof(int));
    bool *visited = malloc(m->nr_nodes * sizeof(bool));
    int *currentBestPath = malloc(m->nr_nodes * sizeof(int)); // This will store the best path temporarily

    if (!travelTimes || !predecessors || !visited || !currentBestPath) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(travelTimes); free(predecessors); free(visited); free(currentBestPath);
        return;
    }

    int currentBestPathSize = 0;
    double bestPathTime = DBL_MAX;
    MinHeap *heap = create_min_heap(m->nr_nodes);
    if (!heap) {
        fprintf(stderr, "Heap allocation failed.\n");
        free(travelTimes); free(predecessors); free(visited); free(currentBestPath);
        return;
    }

    // Initialization
    for (int i = 0; i < m->nr_nodes; i++) {
        travelTimes[i] = DBL_MAX;
        predecessors[i] = -1;
        visited[i] = false;
        currentBestPath[i] = -1;
        if (i == start_id) {
            min_heap_insert(heap, i, 0.0);
            travelTimes[i] = 0.0;
        } else {
            min_heap_insert(heap, i, DBL_MAX);
        }
    }

    while (!is_min_heap_empty(heap)) {
        HeapNode currentNode = min_heap_extract_min(heap);
        int currentNodeId = currentNode.node_id;

        // Bounds check
        if (currentNodeId < 0 || currentNodeId >= m->nr_nodes) {
            fprintf(stderr, "Corrupted node id in heap: %d\n", currentNodeId);
            break;
        }

        if (visited[currentNodeId]) continue;
        visited[currentNodeId] = true;

        // Terminate early if we reach the end node
        if (currentNodeId == end_id) {
            break;
        }

        struct node *currentNodePtr = &m->nodes[currentNodeId];

        for (int i = 0; i < currentNodePtr->num_ways; i++) {
            int way_id = currentNodePtr->way_ids[i];
            // Bounds check for way_id
            if (way_id < 0 || way_id >= m->nr_ways) continue;
            struct way *wayPtr = &m->ways[way_id];

            for (int j = 0; j < wayPtr->num_nodes; j++) {
                int neighborNodeId = wayPtr->node_ids[j];
                // Bounds check for neighborNodeId
                if (neighborNodeId < 0 || neighborNodeId >= m->nr_nodes) continue;
                if (visited[neighborNodeId]) continue;

                double travelTime = calculate_travel_time(*currentNodePtr, m->nodes[neighborNodeId], wayPtr->speed_limit);
                if (travelTimes[currentNodeId] + travelTime < travelTimes[neighborNodeId]) {
                    travelTimes[neighborNodeId] = travelTimes[currentNodeId] + travelTime;
                    predecessors[neighborNodeId] = currentNodeId;
                    min_heap_decrease_key(heap, neighborNodeId, travelTimes[neighborNodeId]);
                }
            }
        }
    }

    // Reconstruct the path from end_id to start_id
    int tempPath[m->nr_nodes];
    int tempPathSize = 0;
    int at = end_id;
    while (at != -1) {
        // Bounds check for at
        if (at < 0 || at >= m->nr_nodes || tempPathSize >= m->nr_nodes) {
            fprintf(stderr, "Path reconstruction failed (cycle or out-of-bounds).\n");
            tempPathSize = 0;
            break;
        }
        tempPath[tempPathSize++] = at;
        at = predecessors[at];
    }

    // Calculate the travel time for this path
    if (tempPathSize > 0) {
        double tempPathTime = ssmap_path_travel_time(m, tempPathSize, tempPath);
        if (tempPathTime < bestPathTime) {
            bestPathTime = tempPathTime;
            currentBestPathSize = tempPathSize;
            for (int i = 0; i < tempPathSize; i++) {
                currentBestPath[i] = tempPath[tempPathSize - 1 - i]; // Reverse the path
            }
        }
    }

    // Print the best path
    for (int i = 0; i < currentBestPathSize; i++) {
        printf("%d ", currentBestPath[i]);
    }
    printf("\n");

    // Cleanup
    free(travelTimes);
    free(predecessors);
    free(visited);
    free(currentBestPath);
    free_min_heap(heap);
}