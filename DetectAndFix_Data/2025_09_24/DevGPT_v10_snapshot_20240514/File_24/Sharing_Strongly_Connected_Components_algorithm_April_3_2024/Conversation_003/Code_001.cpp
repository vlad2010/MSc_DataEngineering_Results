// Assume definitions and setup as in the provided code

void strongConnectIterative(NodeType startNode) {
    std::vector<std::tuple<NodeType, LinkType, bool>> stack; // Node, CurrentLink, IsFirstVisit
    stack.emplace_back(startNode, UNDEFINED_VALUE(LinkType), true);

    while (!stack.empty()) {
        auto& [v, currentLink, isFirstVisit] = stack.back();
        if (isFirstVisit) {
            // Initial visit actions (similar to the start of the recursive function)
            indices[v] = {currIndex, currIndex};
            ++currIndex;
            nodeStack.emplace_back(v);
            assert(!onStackFlag[v]);
            onStackFlag[v] = true;
            isFirstVisit = false; // Mark as visited
            currentLink = link1[v]; // Initialize current link for this node
        } else {
            // Processing neighbors or finishing up
            if (IsDefined(currentLink)) {
                NodeType w = node2Data[currentLink];
                if (!IsDefined(indices[w].first)) {
                    // Neighbor not visited, push to stack for processing
                    stack.emplace_back(w, UNDEFINED_VALUE(LinkType), true);
                } else if (onStackFlag[w]) {
                    // Adjust indices as in the recursive version
                }
                // Move to the next link
                currentLink = nextLink1[currentLink];
            } else {
                // Final actions (similar to the end of the recursive function)
                stack.pop_back();
                // Pop operation, set onStackFlag[w] = false, etc.
            }
        }
    }
}

// Modify the main part of the algorithm to call strongConnectIterative instead of the recursive version
for (NodeType v = 0; v != nrV; ++v) {
    if (!IsDefined(indices[v].first))
        strongConnectIterative(v);
}