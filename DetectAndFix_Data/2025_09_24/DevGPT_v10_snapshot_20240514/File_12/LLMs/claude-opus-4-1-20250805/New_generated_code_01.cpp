#include <graphviz/cgraph.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int main() {
    Agraph_t* graph = nullptr;
    FILE* file = nullptr;
    int ret = EXIT_SUCCESS;
    
    // Create a new directed graph with error checking
    graph = agopen("FSM", Agdirected, nullptr);
    if (!graph) {
        fprintf(stderr, "Error: Failed to create graph\n");
        return EXIT_FAILURE;
    }
    
    // Create nodes for states with error checking
    Agnode_t* state1 = agnode(graph, "State1", true);
    if (!state1) {
        fprintf(stderr, "Error: Failed to create State1 node\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    Agnode_t* state2 = agnode(graph, "State2", true);
    if (!state2) {
        fprintf(stderr, "Error: Failed to create State2 node\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    Agnode_t* state3 = agnode(graph, "State3", true);
    if (!state3) {
        fprintf(stderr, "Error: Failed to create State3 node\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    // Create edges for state transitions with error checking
    Agedge_t* transition1 = agedge(graph, state1, state2, nullptr, true);
    if (!transition1) {
        fprintf(stderr, "Error: Failed to create transition from State1 to State2\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    Agedge_t* transition2 = agedge(graph, state2, state3, nullptr, true);
    if (!transition2) {
        fprintf(stderr, "Error: Failed to create transition from State2 to State3\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    Agedge_t* transition3 = agedge(graph, state3, state1, nullptr, true);
    if (!transition3) {
        fprintf(stderr, "Error: Failed to create transition from State3 to State1\n");
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    // Set attributes for edges with error checking
    if (agsafeset(transition1, "label", "Event1", "") != 0) {
        fprintf(stderr, "Warning: Failed to set label for transition1\n");
    }
    
    if (agsafeset(transition2, "label", "Event2", "") != 0) {
        fprintf(stderr, "Warning: Failed to set label for transition2\n");
    }
    
    if (agsafeset(transition3, "label", "Event3", "") != 0) {
        fprintf(stderr, "Warning: Failed to set label for transition3\n");
    }
    
    // Save the graph to a DOT file with proper error handling
    const char* filename = "fsm.dot";
    file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file '%s' for writing: %s\n", 
                filename, strerror(errno));
        agclose(graph);
        return EXIT_FAILURE;
    }
    
    // Write graph to file with error checking
    if (agwrite(graph, file) != 0) {
        fprintf(stderr, "Error: Failed to write graph to file '%s'\n", filename);
        ret = EXIT_FAILURE;
    }
    
    // Ensure file is properly closed
    if (fclose(file) != 0) {
        fprintf(stderr, "Error: Failed to close file '%s': %s\n", 
                filename, strerror(errno));
        ret = EXIT_FAILURE;
    }
    
    // Cleanup graph resources
    if (agclose(graph) != 0) {
        fprintf(stderr, "Warning: Error during graph cleanup\n");
        // Don't fail the program for cleanup errors, but log them
    }
    
    return ret;
}