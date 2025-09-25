#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include "minheap.h" // Assume MinHeap functions are correctly implemented in this header

// Structures for ssmap, node, way, and MinHeap are assumed to be defined elsewhere

double calculate_travel_time(struct node node1, struct node node2, double speed_limit); // Placeholder
double ssmap_path_travel_time(const struct ssmap * m, int size, int node_ids[]); // Placeholder

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    double distances[m->nr_nodes];
    int predecessors[m->nr_nodes];
    bool visited[m->nr_nodes];
    MinHeap *heap = create_min_heap(m->nr_nodes);

    // Initialize distances and predecessors
    for (int i = 0; i < m->nr_nodes; ++i) {
        distances[i] = DBL_MAX;
        predecessors[i] = -1;
        visited[i] = false;
        min_heap_insert(heap, i, distances[i]);
    }

    distances[start_id] = 0;
    min_heap_decrease_key(heap, start_id, 0);

    while (!is_min_heap_empty(heap)) {
        int currentNode = min_heap_extract_min(heap).node_id;
        visited[currentNode] = true;

        if (currentNode == end_id) {
            break; // Early termination if we reach the destination
        }

        // Iterate over all neighbors of the currentNode
        for (int i = 0; i < m->nodes[currentNode].num_ways; ++i) {
            struct way *currentWay = &m->ways[m->nodes[currentNode].way_ids[i]];
            for (int j = 0; j < currentWay->num_nodes; ++j) {
                int neighborNode = currentWay->node_ids[j];
                if (visited[neighborNode]) continue;

                double travelTime = calculate_travel_time(m->nodes[currentNode], m->nodes[neighborNode], currentWay->speed_limit);
                if (distances[currentNode] + travelTime < distances[neighborNode]) {
                    distances[neighborNode] = distances[currentNode] + travelTime;
                    predecessors[neighborNode] = currentNode;
                    min_heap_decrease_key(heap, neighborNode, distances[neighborNode]);
                }
            }
        }
    }

    // Reconstruct the shortest path
    int path[m->nr_nodes], pathSize = 0;
    for (int at = end_id; at != -1; at = predecessors[at]) {
        path[pathSize++] = at;
    }

    // Optionally, verify the path with ssmap_path_travel_time before printing
    double pathTime = ssmap_path_travel_time(m, pathSize, path);
    printf("Shortest path time: %f\n", pathTime);

    for (int i = pathSize - 1; i >= 0; --i) {
        printf("%d ", path[i]);
    }
    printf("\n");
}