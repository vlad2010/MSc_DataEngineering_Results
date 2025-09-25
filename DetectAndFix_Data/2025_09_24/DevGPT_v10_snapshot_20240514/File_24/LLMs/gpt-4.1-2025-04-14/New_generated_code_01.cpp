void strongConnectIterative(
    const std::vector<NodeType>& node1Data,
    const std::vector<NodeType>& node2Data,
    const std::vector<LinkType>& link1,
    const std::vector<LinkType>& nextLink1,
    std::vector<Couple<NodeType>>& indices,
    NodeType& currIndex,
    sequence_traits<Bool>::container_type& onStackFlag,
    NodeStackType& nodeStack,
    std::vector<PartType>& resSubData,
    PartType& nrParts
) {
    struct Frame {
        NodeType node;
        LinkType currentLink;
        bool processSuccessor;
    };

    std::vector<Frame> stack;

    const size_t n = node1Data.size();
    const size_t indicesSize = indices.size();
    const size_t link1Size = link1.size();
    const size_t node2DataSize = node2Data.size();
    const size_t nextLink1Size = nextLink1.size();
    const size_t onStackFlagSize = onStackFlag.size();
    const size_t resSubDataSize = resSubData.size();

    for (NodeType startNode = 0; startNode < n; ++startNode) {
        // Bounds check for all vectors
        if (static_cast<size_t>(startNode) >= indicesSize ||
            static_cast<size_t>(startNode) >= link1Size ||
            static_cast<size_t>(startNode) >= onStackFlagSize) {
            // Handle error: invalid input data
            throw std::out_of_range("startNode index out of bounds");
        }

        if (!IsDefined(indices[startNode].first)) {
            indices[startNode] = {currIndex, currIndex};

            // Overflow check for currIndex
            if (currIndex == std::numeric_limits<NodeType>::max()) {
                throw std::overflow_error("currIndex overflow");
            }
            ++currIndex;

            nodeStack.push_back(startNode);
            onStackFlag[startNode] = true;

            LinkType startLink = link1[startNode];
            if (static_cast<size_t>(startLink) >= link1Size) {
                throw std::out_of_range("link1[startNode] out of bounds");
            }
            stack.push_back({startNode, startLink, false});

            while (!stack.empty()) {
                auto& frame = stack.back();
                auto& v = frame.node;
                auto& currentLink = frame.currentLink;
                auto& processSuccessor = frame.processSuccessor;

                // Bounds check for v
                if (static_cast<size_t>(v) >= indicesSize ||
                    static_cast<size_t>(v) >= onStackFlagSize) {
                    throw std::out_of_range("v index out of bounds");
                }

                if (processSuccessor) {
                    if (static_cast<size_t>(currentLink) >= node2DataSize ||
                        static_cast<size_t>(currentLink) >= indicesSize) {
                        throw std::out_of_range("currentLink out of bounds in processSuccessor");
                    }
                    NodeType w = node2Data[currentLink];
                    if (static_cast<size_t>(w) >= indicesSize) {
                        throw std::out_of_range("w out of bounds in processSuccessor");
                    }
                    indices[v].second = std::min(indices[v].second, indices[w].second);
                    processSuccessor = false; // Reset flag
                    if (static_cast<size_t>(currentLink) >= nextLink1Size) {
                        throw std::out_of_range("currentLink out of bounds in nextLink1");
                    }
                    currentLink = nextLink1[currentLink]; // Move to next link after processing
                }

                if (IsDefined(currentLink)) {
                    if (static_cast<size_t>(currentLink) >= node2DataSize) {
                        throw std::out_of_range("currentLink out of bounds in node2Data");
                    }
                    NodeType w = node2Data[currentLink];
                    if (static_cast<size_t>(w) >= indicesSize ||
                        static_cast<size_t>(w) >= link1Size ||
                        static_cast<size_t>(w) >= onStackFlagSize) {
                        throw std::out_of_range("w out of bounds in successors");
                    }
                    if (!IsDefined(indices[w].first)) {
                        processSuccessor = true; // Indicate to process the successor next
                        LinkType wLink = link1[w];
                        if (static_cast<size_t>(wLink) >= link1Size) {
                            throw std::out_of_range("link1[w] out of bounds");
                        }
                        stack.push_back({w, wLink, false}); // Node w has not been visited, start DFS from w
                    } else {
                        if (onStackFlag[w]) {
                            // Only minimize indices[v].second with indices[w].first if w is on the stack
                            indices[v].second = std::min(indices[v].second, indices[w].first);
                        }
                        if (static_cast<size_t>(currentLink) >= nextLink1Size) {
                            throw std::out_of_range("currentLink out of bounds in nextLink1");
                        }
                        currentLink = nextLink1[currentLink]; // Move to next link; already reset processSuccessor
                    }
                } else {
                    // No more successors to process, check if we've completed a strongly connected component
                    if (indices[v].first == indices[v].second) {
                        NodeType w;
                        do {
                            if (nodeStack.empty()) {
                                throw std::runtime_error("nodeStack underflow");
                            }
                            w = nodeStack.back(); nodeStack.pop_back();
                            if (static_cast<size_t>(w) >= onStackFlagSize ||
                                static_cast<size_t>(w) >= resSubDataSize) {
                                throw std::out_of_range("w out of bounds in SCC");
                            }
                            onStackFlag[w] = false;
                            resSubData[w] = nrParts;
                        } while (w != v);

                        // Overflow check for nrParts
                        if (nrParts == std::numeric_limits<PartType>::max()) {
                            throw std::overflow_error("nrParts overflow");
                        }
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