#include <graphviz/cgraph.h>
#include <cstdio>
#include <cstdlib>

int main() {
    // Create a new directed graph
    Agraph_t* graph = agopen("FSM", Agdirected, nullptr);
    if (!graph) {
        fprintf(stderr, "Error: Failed to create graph.\n");
        return EXIT_FAILURE;
    }

    // Create nodes for states
    Agnode_t* state1 = agnode(graph, "State1", true);
    if (!state1) {
        fprintf(stderr, "Error: Failed to create node State1.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    Agnode_t* state2 = agnode(graph, "State2", true);
    if (!state2) {
        fprintf(stderr, "Error: Failed to create node State2.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    Agnode_t* state3 = agnode(graph, "State3", true);
    if (!state3) {
        fprintf(stderr, "Error: Failed to create node State3.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }

    // Create edges for state transitions
    Agedge_t* transition1 = agedge(graph, state1, state2, nullptr, true);
    if (!transition1) {
        fprintf(stderr, "Error: Failed to create edge State1->State2.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    Agedge_t* transition2 = agedge(graph, state2, state3, nullptr, true);
    if (!transition2) {
        fprintf(stderr, "Error: Failed to create edge State2->State3.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    Agedge_t* transition3 = agedge(graph, state3, state1, nullptr, true);
    if (!transition3) {
        fprintf(stderr, "Error: Failed to create edge State3->State1.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }

    // Set attributes for edges
    agsafeset(transition1, "label", "Event1", "");
    agsafeset(transition2, "label", "Event2", "");
    agsafeset(transition3, "label", "Event3", "");

    // Save the graph to a DOT file
    FILE* file = fopen("fsm.dot", "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file for writing.\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    if (agwrite(graph, file) != 0) {
        fprintf(stderr, "Error: Failed to write graph to file.\n");
        fclose(file);
        agclose(graph);
        return EXIT_FAILURE;
    }
    fclose(file);

    // Cleanup
    agclose(graph);

    return EXIT_SUCCESS;
}