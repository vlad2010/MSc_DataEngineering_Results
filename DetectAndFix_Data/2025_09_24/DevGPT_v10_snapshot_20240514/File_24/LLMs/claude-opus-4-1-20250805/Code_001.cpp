void strongConnectIterative(const std::vector<NodeType>& node1Data, const std::vector<NodeType>& node2Data, const std::vector<LinkType>& link1, const std::vector<LinkType>& nextLink1, std::vector<Couple<NodeType>>& indices, NodeType& currIndex, sequence_traits<Bool>::container_type& onStackFlag, NodeStackType& nodeStack, std::vector<PartType>& resSubData, PartType& nrParts) {
    struct Frame {
        NodeType node;
        LinkType currentLink;
        bool processSuccessor;
    };

    std::vector<Frame> stack;

    for (NodeType startNode = 0; startNode < node1Data.size(); ++startNode) {
        if (!IsDefined(indices[startNode].first)) {
            indices[startNode] = {currIndex, currIndex};
            ++currIndex;
            nodeStack.push_back(startNode);
            onStackFlag[startNode] = true;

            stack.push_back({startNode, link1[startNode], false});

            while (!stack.empty()) {
                auto& frame = stack.back();
                auto& v = frame.node;
                auto& currentLink = frame.currentLink;
                auto& processSuccessor = frame.processSuccessor;

                if (processSuccessor) {
                    NodeType w = node2Data[currentLink];
                    indices[v].second = std::min(indices[v].second, indices[w].second);
                    processSuccessor = false; // Reset flag
                    currentLink = nextLink1[currentLink]; // Move to next link after processing
                }

                if (IsDefined(currentLink)) {
                    NodeType w = node2Data[currentLink];
                    if (!IsDefined(indices[w].first)) {
                        processSuccessor = true; // Indicate to process the successor next
                        stack.push_back({w, link1[w], false}); // Node w has not been visited, start DFS from w
                    } else {
                        if (onStackFlag[w]) {
                            // Only minimize indices[v].second with indices[w].first if w is on the stack
                            indices[v].second = std::min(indices[v].second, indices[w].first);
                        }
                        currentLink = nextLink1[currentLink]; // Move to next link; already reset processSuccessor
                    }
                } else {
                    // No more successors to process, check if we've completed a strongly connected component
                    if (indices[v].first == indices[v].second) {
                        NodeType w;
                        do {
                            w = nodeStack.back(); nodeStack.pop_back();
                            onStackFlag[w] = false;
                            resSubData[w] = nrParts;
                        } while (w != v);
                        nrParts++;
                    }
                    stack.pop_back(); // Finish processing node v
                }

                // Ensure the current frame is updated correctly
                frame.currentLink = currentLink;
                frame.processSuccessor = processSuccessor;
            }
        }
    }
}