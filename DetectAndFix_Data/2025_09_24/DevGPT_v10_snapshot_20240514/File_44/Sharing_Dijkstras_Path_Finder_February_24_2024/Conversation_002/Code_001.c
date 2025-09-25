void ssmap_path_create(const struct ssmap *m, int start_id, int end_id) {
    double *distances = malloc(m->nr_nodes * sizeof(double));
    int *predecessors = malloc(m->nr_nodes * sizeof(int));
    bool *visited = malloc(m->nr_nodes * sizeof(bool));
    MinHeap *heap = create_min_heap(m->nr_nodes);

    for (int i = 0; i < m->nr_nodes; i++) {
        distances[i] = DBL_MAX;
        predecessors[i] = -1;
        visited[i] = false;
        min_heap_insert(heap, i, distances[i]);
    }

    distances[start_id] = 0;
    min_heap_decrease_key(heap, start_id, 0);

    while (!is_min_heap_empty(heap)) {
        HeapNode currentNode = min_heap_extract_min(heap);
        visited[currentNode.node_id] = true;

        if (currentNode.node_id == end_id) break; // Stop if we reach the end node

        for (int i = 0; i < m->nodes[currentNode.node_id].num_ways; i++) {
            int wayIndex = m->nodes[currentNode.node_id].way_ids[i];
            struct way currentWay = m->ways[wayIndex];
            for (int j = 0; j < currentWay.num_nodes; j++) {
                int neighborNode = currentWay.node_ids[j];

                if (!visited[neighborNode]) {
                    double travelTime = calculate_travel_time(m->nodes[currentNode.node_id], m->nodes[neighborNode], currentWay.speed_limit);
                    double newDistance = distances[currentNode.node_id] + travelTime;

                    if (newDistance < distances[neighborNode]) {
                        distances[neighborNode] = newDistance;
                        predecessors[neighborNode] = currentNode.node_id;
                        min_heap_decrease_key(heap, neighborNode, newDistance);
                    }
                }
            }
        }
    }

    // Print the path from start_id to end_id by backtracking
    int current = end_id;
    int pathSize = 0;
    while (current != -1 && pathSize < m->nr_nodes) {
        printf("%d ", current);
        current = predecessors[current];
        pathSize++;
    }

    free(distances);
    free(predecessors);
    free(visited);
    // Free the heap and its elements here as well
}