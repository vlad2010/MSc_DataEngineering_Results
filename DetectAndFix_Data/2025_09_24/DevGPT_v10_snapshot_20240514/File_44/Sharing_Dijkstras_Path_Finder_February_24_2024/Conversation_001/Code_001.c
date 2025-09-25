#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h> // For DBL_MAX

// Definitions for node, way, and ssmap structs as provided.

typedef struct node {
    double lat;
    double lon;
    int id;
    int osmid;
    int num_ways;
    int *way_ids;
} node;

typedef struct way {
    int id;
    int osmid;
    char *name;
    float speed_limit;
    bool one_way;
    int num_nodes;
    int *node_ids;
} way;

typedef struct ssmap {
    int nr_nodes;
    int nr_ways;
    struct node *nodes;
    struct way *ways;
} ssmap;

// Definitions for MinHeap and HeapNode as provided.

// Function prototypes for min heap operations and travel time calculations
// Assuming these are defined elsewhere.
MinHeap *create_min_heap(int capacity);
void min_heap_insert(MinHeap *heap, int node_id, double priority);
HeapNode min_heap_extract_min(MinHeap *heap);
void min_heap_decrease_key(MinHeap *heap, int node_id, double priority);
bool is_min_heap_empty(MinHeap *heap);
double calculate_travel_time(struct node node1, struct node node2, double speed_limit);
double ssmap_path_travel_time(const struct ssmap *m, int size, int node_ids[]);

void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    double distances[m->nr_nodes];
    int predecessors[m->nr_nodes];
    bool visited[m->nr_nodes];
    for(int i = 0; i < m->nr_nodes; i++) {
        distances[i] = DBL_MAX;
        predecessors[i] = -1;
        visited[i] = false;
    }

    distances[start_id] = 0;
    MinHeap *heap = create_min_heap(m->nr_nodes);
    min_heap_insert(heap, start_id, 0);

    while(!is_min_heap_empty(heap)) {
        HeapNode minNode = min_heap_extract_min(heap);
        int currentNode = minNode.node_id;
        if(visited[currentNode]) continue;
        visited[currentNode] = true;

        struct node currentNodeStruct = m->nodes[currentNode];
        for(int i = 0; i < currentNodeStruct.num_ways; i++) {
            struct way currentWay = m->ways[currentNodeStruct.way_ids[i]];
            for(int j = 0; j < currentWay.num_nodes; j++) {
                int neighborNode = currentWay.node_ids[j];
                if(visited[neighborNode]) continue;
                double travelTime = calculate_travel_time(currentNodeStruct, m->nodes[neighborNode], currentWay.speed_limit);
                double newDistance = distances[currentNode] + travelTime;
                if(newDistance < distances[neighborNode]) {
                    distances[neighborNode] = newDistance;
                    predecessors[neighborNode] = currentNode;
                    min_heap_insert(heap, neighborNode, newDistance);
                }
            }
        }
    }

    // Backtrack from end_id to start_id to print the shortest path
    int path[m->nr_nodes];
    int pathSize = 0;
    for(int at = end_id; at != -1; at = predecessors[at]) {
        path[pathSize++] = at;
    }

    for(int i = pathSize - 1; i >= 0; i--) {
        printf("%d ", path[i]);
    }
    printf("\n");
}