#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

// Placeholder function signatures
double calculate_travel_time(struct node node1, struct node node2, double speed_limit);
double ssmap_path_travel_time(const struct ssmap * m, int size, int node_ids[]);
void min_heap_insert(MinHeap *heap, int node_id, double priority);
HeapNode min_heap_extract_min(MinHeap *heap);
bool is_min_heap_empty(MinHeap *heap);
MinHeap* create_min_heap(int capacity);
void min_heap_decrease_key(MinHeap *heap, int node_id, double priority);
void free_min_heap(MinHeap *heap);

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    // Input validation
    if (m == NULL || m->nodes == NULL || m->ways == NULL) {
        fprintf(stderr, "Error: Invalid map structure\n");
        return;
    }
    
    if (m->nr_nodes <= 0 || m->nr_nodes > INT_MAX / 4) {
        fprintf(stderr, "Error: Invalid number of nodes\n");
        return;
    }
    
    if (start_id < 0 || start_id >= m->nr_nodes || 
        end_id < 0 || end_id >= m->nr_nodes) {
        fprintf(stderr, "Error: Invalid start or end node ID\n");
        return;
    }
    
    // Check for integer overflow in allocation size
    size_t alloc_size = 0;
    if (__builtin_mul_overflow(m->nr_nodes, sizeof(double), &alloc_size)) {
        fprintf(stderr, "Error: Allocation size overflow\n");
        return;
    }
    
    // Initial setup with NULL checks
    double *travelTimes = calloc(m->nr_nodes, sizeof(double));
    if (travelTimes == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for travelTimes\n");
        return;
    }
    
    int *predecessors = calloc(m->nr_nodes, sizeof(int));
    if (predecessors == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for predecessors\n");
        free(travelTimes);
        return;
    }
    
    bool *visited = calloc(m->nr_nodes, sizeof(bool));
    if (visited == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for visited\n");
        free(travelTimes);
        free(predecessors);
        return;
    }
    
    int *currentBestPath = calloc(m->nr_nodes, sizeof(int));
    if (currentBestPath == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for currentBestPath\n");
        free(travelTimes);
        free(predecessors);
        free(visited);
        return;
    }
    
    int *tempPath = calloc(m->nr_nodes, sizeof(int));
    if (tempPath == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for tempPath\n");
        free(travelTimes);
        free(predecessors);
        free(visited);
        free(currentBestPath);
        return;
    }
    
    int currentBestPathSize = 0;
    double bestPathTime = DBL_MAX;
    MinHeap *heap = create_min_heap(m->nr_nodes);
    if (heap == NULL) {
        fprintf(stderr, "Error: Failed to create min heap\n");
        free(travelTimes);
        free(predecessors);
        free(visited);
        free(currentBestPath);
        free(tempPath);
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
        
        // Validate node ID
        if (currentNodeId < 0 || currentNodeId >= m->nr_nodes) {
            continue;
        }
        
        visited[currentNodeId] = true;

        // Terminate early if we reach the end node
        if (currentNodeId == end_id) {
            break;
        }

        struct node *currentNodePtr = &m->nodes[currentNodeId];
        if (currentNodePtr == NULL || currentNodePtr->way_ids == NULL) {
            continue;
        }
        
        for (int i = 0; i < currentNodePtr->num_ways && i < INT_MAX; i++) {
            // Bounds check for way_ids
            int way_id = currentNodePtr->way_ids[i];
            if (way_id < 0 || way_id >= m->nr_ways) {
                continue;
            }
            
            struct way *wayPtr = &m->ways[way_id];
            if (wayPtr == NULL || wayPtr->node_ids == NULL) {
                continue;
            }
            
            for (int j = 0; j < wayPtr->num_nodes && j < INT_MAX; j++) {
                int neighborNodeId = wayPtr->node_ids[j];
                
                // Validate neighbor node ID
                if (neighborNodeId < 0 || neighborNodeId >= m->nr_nodes) {
                    continue;
                }
                
                if (visited[neighborNodeId]) continue;

                double travelTime = calculate_travel_time(*currentNodePtr, m->nodes[neighborNodeId], wayPtr->speed_limit);
                
                // Check for overflow in addition
                if (travelTimes[currentNodeId] != DBL_MAX && 
                    travelTime != DBL_MAX &&
                    travelTimes[currentNodeId] + travelTime < travelTimes[neighborNodeId]) {
                    travelTimes[neighborNodeId] = travelTimes[currentNodeId] + travelTime;
                    predecessors[neighborNodeId] = currentNodeId;
                    min_heap_decrease_key(heap, neighborNodeId, travelTimes[neighborNodeId]);
                }
            }
        }
    }

    // Reconstruct the path from end_id to start_id
    int tempPathSize = 0;
    for (int at = end_id; at != -1 && tempPathSize < m->nr_nodes; at = predecessors[at]) {
        tempPath[tempPathSize++] = at;
        // Prevent infinite loops
        if (tempPathSize >= m->nr_nodes) {
            fprintf(stderr, "Error: Path reconstruction failed - possible cycle\n");
            break;
        }
    }

    // Calculate the travel time for this path
    if (tempPathSize > 0) {
        double tempPathTime = ssmap_path_travel_time(m, tempPathSize, tempPath);
        if (tempPathTime < bestPathTime) {
            bestPathTime = tempPathTime;
            currentBestPathSize = tempPathSize;
            for (int i = 0; i < tempPathSize && i < m->nr_nodes; i++) {
                currentBestPath[i] = tempPath[tempPathSize - 1 - i]; // Reverse the path
            }
        }
    }

    // Print the best path
    for (int i = 0; i < currentBestPathSize && i < m->nr_nodes; i++) {
        printf("%d ", currentBestPath[i]);
    }
    if (currentBestPathSize > 0) {
        printf("\n");
    }

    // Cleanup
    free(travelTimes);
    free(predecessors);
    free(visited);
    free(currentBestPath);
    free(tempPath);
    free_min_heap(heap);
}