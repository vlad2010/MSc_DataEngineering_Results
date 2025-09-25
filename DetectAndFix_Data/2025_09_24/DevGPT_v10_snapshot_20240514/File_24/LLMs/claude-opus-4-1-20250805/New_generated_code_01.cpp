#include <limits>
#include <stdexcept>

void strongConnectIterative(const std::vector<NodeType>& node1Data, 
                           const std::vector<NodeType>& node2Data, 
                           const std::vector<LinkType>& link1, 
                           const std::vector<LinkType>& nextLink1, 
                           std::vector<Couple<NodeType>>& indices, 
                           NodeType& currIndex, 
                           sequence_traits<Bool>::container_type& onStackFlag, 
                           NodeStackType& nodeStack, 
                           std::vector<PartType>& resSubData, 
                           PartType& nrParts) {
    
    // Input validation
    if (node1Data.empty() || indices.size() != node1Data.size() || 
        onStackFlag.size() != node1Data.size() || 
        resSubData.size() != node1Data.size() ||
        link1.size() != node1Data.size()) {
        throw std::invalid_argument("Invalid input vector sizes");
    }
    
    // Validate that node2Data and nextLink1 have consistent sizes
    if (!node2Data.empty() && !nextLink1.empty() && node2Data.size() != nextLink1.size()) {
        throw std::invalid_argument("Inconsistent link data sizes");
    }
    
    struct Frame {
        NodeType node;
        LinkType currentLink;
        bool processSuccessor;
    };

    std::vector<Frame> stack;
    stack.reserve(node1Data.size()); // Pre-allocate to avoid reallocation

    for (NodeType startNode = 0; startNode < node1Data.size(); ++startNode) {
        if (!IsDefined(indices[startNode].first)) {
            // Check for integer overflow
            if (currIndex == std::numeric_limits<NodeType>::max()) {
                throw std::overflow_error("currIndex overflow");
            }
            
            indices[startNode] = {currIndex, currIndex};
            ++currIndex;
            nodeStack.push_back(startNode);
            onStackFlag[startNode] = true;

            stack.push_back({startNode, link1[startNode], false});

            while (!stack.empty()) {
                // Use index instead of reference to avoid invalidation
                size_t frameIndex = stack.size() - 1;
                auto v = stack[frameIndex].node;
                auto currentLink = stack[frameIndex].currentLink;
                auto processSuccessor = stack[frameIndex].processSuccessor;

                if (processSuccessor) {
                    // Bounds check before accessing node2Data
                    if (currentLink >= node2Data.size()) {
                        throw std::out_of_range("currentLink out of bounds for node2Data");
                    }
                    NodeType w = node2Data[currentLink];
                    
                    // Bounds check for indices
                    if (v >= indices.size() || w >= indices.size()) {
                        throw std::out_of_range("Node index out of bounds");
                    }
                    
                    indices[v].second = std::min(indices[v].second, indices[w].second);
                    processSuccessor = false;
                    
                    // Bounds check before accessing nextLink1
                    if (currentLink >= nextLink1.size()) {
                        throw std::out_of_range("currentLink out of bounds for nextLink1");
                    }
                    currentLink = nextLink1[currentLink];
                }

                if (IsDefined(currentLink)) {
                    // Bounds check before accessing node2Data
                    if (currentLink >= node2Data.size()) {
                        throw std::out_of_range("currentLink out of bounds for node2Data");
                    }
                    NodeType w = node2Data[currentLink];
                    
                    // Bounds check for w
                    if (w >= indices.size()) {
                        throw std::out_of_range("Node w out of bounds");
                    }
                    
                    if (!IsDefined(indices[w].first)) {
                        processSuccessor = true;
                        
                        // Bounds check before accessing link1
                        if (w >= link1.size()) {
                            throw std::out_of_range("Node w out of bounds for link1");
                        }
                        
                        // Check stack size to prevent excessive memory usage
                        if (stack.size() >= node1Data.size() * 2) {
                            throw std::runtime_error("Stack depth exceeded reasonable limit");
                        }
                        
                        stack.push_back({w, link1[w], false});
                    } else {
                        // Bounds check for onStackFlag
                        if (w >= onStackFlag.size()) {
                            throw std::out_of_range("Node w out of bounds for onStackFlag");
                        }
                        
                        if (onStackFlag[w]) {
                            indices[v].second = std::min(indices[v].second, indices[w].first);
                        }
                        
                        // Bounds check before accessing nextLink1
                        if (currentLink >= nextLink1.size()) {
                            throw std::out_of_range("currentLink out of bounds for nextLink1");
                        }
                        currentLink = nextLink1[currentLink];
                    }
                } else {
                    // Bounds check for v
                    if (v >= indices.size()) {
                        throw std::out_of_range("Node v out of bounds");
                    }
                    
                    if (indices[v].first == indices[v].second) {
                        // Check for integer overflow
                        if (nrParts == std::numeric_limits<PartType>::max()) {
                            throw std::overflow_error("nrParts overflow");
                        }
                        
                        NodeType w;
                        do {
                            if (nodeStack.empty()) {
                                throw std::runtime_error("nodeStack unexpectedly empty");
                            }
                            w = nodeStack.back(); 
                            nodeStack.pop_back();
                            
                            // Bounds check
                            if (w >= onStackFlag.size() || w >= resSubData.size()) {
                                throw std::out_of_range("Node w out of bounds in SCC processing");
                            }
                            
                            onStackFlag[w] = false;
                            resSubData[w] = nrParts;
                        } while (w != v && !nodeStack.empty());
                        
                        nrParts++;
                    }
                    stack.pop_back();
                } else {
                    // Update the frame at the current index (safe after potential reallocation)
                    if (frameIndex < stack.size()) {
                        stack[frameIndex].currentLink = currentLink;
                        stack[frameIndex].processSuccessor = processSuccessor;
                    }
                }
            }
        }
    }
}