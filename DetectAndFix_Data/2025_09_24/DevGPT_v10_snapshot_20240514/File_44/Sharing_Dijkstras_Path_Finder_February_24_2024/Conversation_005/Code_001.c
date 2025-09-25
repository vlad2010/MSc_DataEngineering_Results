#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>

// Assume the definitions for `struct ssmap`, `struct node`, `struct way`, and MinHeap functions are provided elsewhere.

// Placeholder function signatures
double calculate_travel_time(struct node node1, struct node node2, double speed_limit);
double ssmap_path_travel_time(const struct ssmap * m, int size, int node_ids[]);
void min_heap_insert(MinHeap *heap, int node_id, double priority);
HeapNode min_heap_extract_min(MinHeap *heap);
bool is_min_heap_empty(MinHeap *heap);
MinHeap* create_min_heap(int capacity);
void min_heap_decrease_key(MinHeap *heap, int node_id, double priority);

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    // Initial setup
    double *travelTimes = malloc(m->nr_nodes * sizeof(double));
    int *predecessors = malloc(m->nr_nodes * sizeof(int));
    bool *visited = malloc(m->nr_nodes * sizeof(bool));
    int *currentBestPath = malloc(m->nr_nodes * sizeof(int)); // This will store the best path temporarily
    int currentBestPathSize = 0;
    double bestPathTime = DBL_MAX;
    MinHeap *heap = create_min_heap(m->nr_nodes);

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
        visited[currentNodeId] = true;

        // Terminate early if we reach the end node
        if (currentNodeId == end_id) {
            break;
        }

        struct node *currentNodePtr = &m->nodes[currentNodeId];
        for (int i = 0; i < currentNodePtr->num_ways; i++) {
            struct way *wayPtr = &m->ways[currentNodePtr->way_ids[i]];
            for (int j = 0; j < wayPtr->num_nodes; j++) {
                int neighborNodeId = wayPtr->node_ids[j];
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
    for (int at = end_id; at != -1; at = predecessors[at]) {
        tempPath[tempPathSize++] = at;
    }

    // Calculate the travel time for this path
    double tempPathTime = ssmap_path_travel_time(m, tempPathSize, tempPath);
    if (tempPathTime < bestPathTime) {
        bestPathTime = tempPathTime;
        currentBestPathSize = tempPathSize;
        for (int i = 0; i < tempPathSize; i++) {
            currentBestPath[i] = tempPath[tempPathSize - 1 - i]; // Reverse the path
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
    // Assume there's a function to free the heap
}
