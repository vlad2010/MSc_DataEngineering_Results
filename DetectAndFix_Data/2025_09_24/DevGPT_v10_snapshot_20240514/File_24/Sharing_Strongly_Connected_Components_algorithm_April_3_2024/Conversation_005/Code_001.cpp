#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>

// Assuming definitions from the provided context...

void strongConnectIterative(const std::vector<NodeType>& node1Data, const std::vector<NodeType>& node2Data, const std::vector<LinkType>& link1, const std::vector<LinkType>& nextLink1, std::vector<Couple<NodeType>>& indices, NodeType& currIndex, sequence_traits<Bool>::container_type& onStackFlag, NodeStackType& nodeStack, std::vector<PartType>& resSubData, PartType& nrParts) {
    struct Frame {
        NodeType node;
        LinkType currentLink;
        bool isFirstVisit;
        bool processSuccessor;
    };
    
    std::vector<Frame> stack;
    stack.push_back({startNode, link1[startNode], true, false});

    while (!stack.empty()) {
        auto& frame = stack.back();
        auto& [v, currentLink, isFirstVisit, processSuccessor] = frame;

        if (isFirstVisit) {
            indices[v] = {currIndex, currIndex};
            ++currIndex;
            nodeStack.push_back(v);
            onStackFlag[v] = true;
            isFirstVisit = false;
        } else if (processSuccessor) {
            // After "recursion", minimize indices[v].second based on the successor's indices
            NodeType w = node2Data[currentLink];
            if (onStackFlag[w]) {
                indices[v].second = std::min(indices[v].second, indices[w].first);
            }
            processSuccessor = false; // Reset flag
            currentLink = nextLink1[currentLink]; // Move to next link after processing
        }

        if (!processSuccessor && IsDefined(currentLink)) {
            NodeType w = node2Data[currentLink];
            if (!IsDefined(indices[w].first)) {
                // Node w has not been visited
                stack.push_back({w, link1[w], true, false});
                processSuccessor = true; // Indicate to process the successor next
            } else {
                if (onStackFlag[w]) {
                    indices[v].second = std::min(indices[v].second, indices[w].first);
                }
                currentLink = nextLink1[currentLink]; // Move to next link
            }
        } else if (!IsDefined(currentLink)) {
            // No more successors to process
            if (indices[v].first == indices[v].second) {
                // Complete a strongly connected component
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