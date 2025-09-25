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
                }

                // Label to return to after potentially modifying the stack,
                // ensuring references are valid.
                CHECK_CURRENT_LINK:
                if (IsDefined(currentLink)) {
                    NodeType w = node2Data[currentLink];
                    if (!IsDefined(indices[w].first)) {
                        // Prepare to process the successor
                        processSuccessor = true;
                        // Modifying the stack, so we need to use goto to ensure the frame reference is updated
                        stack.push_back({w, link1[w], false}); 
                        continue; // This continue works with the goto to process the new frame
                    } else {
                        if (onStackFlag[w]) {
                            indices[v].second = std::min(indices[v].second, indices[w].first);
                        }
                        currentLink = nextLink1[currentLink]; // Move to next link
                        goto CHECK_CURRENT_LINK; // Recheck currentLink with the updated frame reference
                    }
                } else {
                    // Finish processing node v
                    if (indices[v].first == indices[v].second) {
                        NodeType w;
                        do {
                            w = nodeStack.back(); nodeStack.pop_back();
                            onStackFlag[w] = false;
                            resSubData[w] = nrParts;
                        } while (w != v);
                        nrParts++;
                    }
                    stack.pop_back(); // Remove the frame for v
                }
            }
        }
    }
}