#include <vector>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

template <typename NodeType, typename LinkType, typename PartType, typename Bool,
          typename Couple, typename NodeStackType>
void strongConnectIterative(const std::vector<NodeType>& node1Data,
                            const std::vector<NodeType>& node2Data,
                            const std::vector<LinkType>& link1,
                            const std::vector<LinkType>& nextLink1,
                            std::vector<Couple>& indices,
                            NodeType& currIndex,
                            typename /* sequence_traits<Bool>::container_type */ std::vector<Bool>& onStackFlag,
                            NodeStackType& nodeStack,
                            std::vector<PartType>& resSubData,
                            PartType& nrParts)
{
    static_assert(std::is_integral_v<NodeType>, "NodeType must be integral");
    static_assert(std::is_integral_v<LinkType>, "LinkType must be integral");
    static_assert(std::is_integral_v<PartType>, "PartType must be integral");

    // Basic size/coherence checks to avoid UB (CWE-125/787/129).
    const size_t N = node1Data.size();
    if (indices.size() < N || link1.size() < N || onStackFlag.size() < N || resSubData.size() < N) {
        throw std::invalid_argument("Input container sizes are inconsistent with node count");
    }
    if (nextLink1.size() != node2Data.size()) {
        // Each link index points into node2Data and nextLink1 should have same domain
        throw std::invalid_argument("nextLink1 size must equal node2Data size");
    }

    // Helper that computes "invalid" sentinel accounting for signed/unsigned types (CWE-195/190).
    auto invalidLinkValue = []() constexpr -> LinkType {
        if constexpr (std::is_signed_v<LinkType>) {
            return static_cast<LinkType>(-1);
        } else {
            return std::numeric_limits<LinkType>::max();
        }
    };

    // Check if link is defined and in range of node2Data.
    auto linkIsDefined = [&](LinkType l) -> bool {
        // Undefined by sentinel
        if constexpr (std::is_signed_v<LinkType>) {
            if (l < 0) return false;
        } else {
            if (l == std::numeric_limits<LinkType>::max()) return false;
        }
        // Undefined if out of range (defensive)
        size_t idx = static_cast<size_t>(l);
        return idx < node2Data.size();
    };

    // Safe index checks for node indices.
    auto nodeIndexValid = [&](NodeType v) -> bool {
        if constexpr (std::is_signed_v<NodeType>) {
            return v >= 0 && static_cast<size_t>(v) < N;
        } else {
            return static_cast<size_t>(v) < N;
        }
    };

    auto firstLinkOf = [&](NodeType v) -> LinkType {
        if (!nodeIndexValid(v)) return invalidLinkValue();
        return link1[static_cast<size_t>(v)];
    };

    auto nextLinkOf = [&](LinkType l) -> LinkType {
        size_t idx = static_cast<size_t>(l);
        if (idx >= nextLink1.size()) return invalidLinkValue();
        return nextLink1[idx];
    };

    auto nodeFromLink = [&](LinkType l) -> NodeType {
        size_t idx = static_cast<size_t>(l);
        if (idx >= node2Data.size()) {
            throw std::out_of_range("Link index out of range for node2Data");
        }
        return node2Data[idx];
    };

    // IsDefined for indices[...] is assumed available in the caller’s domain.
    // Provide a fallback if Couple supports .first as NodeType and undefined is negative or max.
    auto isIndexDefined = [&](const Couple& c) -> bool {
        // If caller provided IsDefined, prefer that; else heuristic:
        // Defined if c.first is within 0..N-1 range for NodeType.
        if constexpr (std::is_signed_v<NodeType>) {
            return c.first >= 0; // assumes -1 means undefined
        } else {
            // For unsigned NodeType, assume max value means undefined
            return c.first != std::numeric_limits<NodeType>::max();
        }
    };

    struct Frame {
        NodeType node;
        LinkType currentLink;
        bool processSuccessor; // true means: when we come back, update lowlink[v] with lowlink[w]
    };

    std::vector<Frame> stack;
    stack.reserve(N); // avoid reallocation (helps avoid invalidation), but we still avoid references

    for (size_t start = 0; start < N; ++start) {
        NodeType startNode = static_cast<NodeType>(start);
        if (!isIndexDefined(indices[startNode])) {
            // Initialize node startNode
            indices[startNode].first  = currIndex;
            indices[startNode].second = currIndex;
            ++currIndex;
            nodeStack.push_back(startNode);
            onStackFlag[startNode] = true;

            stack.push_back(Frame{ startNode, firstLinkOf(startNode), false });

            while (!stack.empty()) {
                // Work on a copy; write back to stack[i] when mutated to avoid ref invalidation (CWE-416 fix).
                const size_t i = stack.size() - 1;
                Frame fr = stack[i];
                NodeType v = fr.node;

                if (fr.processSuccessor) {
                    // We just returned from exploring successor w from currentLink
                    if (!linkIsDefined(fr.currentLink)) {
                        // Defensive: if currentLink became invalid, skip
                        fr.processSuccessor = false;
                    } else {
                        NodeType w = nodeFromLink(fr.currentLink);
                        if (!nodeIndexValid(w)) {
                            throw std::out_of_range("w out of node range");
                        }
                        // lowlink[v] = min(lowlink[v], lowlink[w])
                        indices[v].second = std::min(indices[v].second, indices[w].second);
                        fr.processSuccessor = false;
                        fr.currentLink = nextLinkOf(fr.currentLink);
                    }
                    stack[i] = fr;
                    continue;
                }

                if (linkIsDefined(fr.currentLink)) {
                    NodeType w = nodeFromLink(fr.currentLink);
                    if (!nodeIndexValid(w)) {
                        throw std::out_of_range("w out of node range");
                    }

                    if (!isIndexDefined(indices[w])) {
                        // Discover w: initialize indices, push to nodeStack and traversal stack (fixes logic bug).
                        indices[w].first  = currIndex;
                        indices[w].second = currIndex;
                        ++currIndex;
                        nodeStack.push_back(w);
                        onStackFlag[w] = true;

                        // On return, update lowlink[v] with lowlink[w]
                        fr.processSuccessor = true;
                        stack[i] = fr;

                        // Continue DFS from w
                        stack.push_back(Frame{ w, firstLinkOf(w), false });
                        continue; // Important: re-acquire top on next iteration
                    } else {
                        if (onStackFlag[w]) {
                            // lowlink[v] = min(lowlink[v], index[w])
                            indices[v].second = std::min(indices[v].second, indices[w].first);
                        }
                        fr.currentLink = nextLinkOf(fr.currentLink);
                        // processSuccessor remains false
                        stack[i] = fr;
                        continue;
                    }
                } else {
                    // No more successors for v: if root of SCC, pop stack until v
                    if (indices[v].first == indices[v].second) {
                        while (true) {
                            if (nodeStack.empty()) {
                                // Defensive: should never happen if logic is correct
                                throw std::runtime_error("nodeStack underflow");
                            }
                            NodeType w = nodeStack.back();
                            nodeStack.pop_back();
                            if (!nodeIndexValid(w)) {
                                throw std::out_of_range("Popped node out of range");
                            }
                            onStackFlag[w] = false;
                            resSubData[w] = nrParts;
                            if (w == v) break;
                        }
                        ++nrParts;
                    }
                    // Done with v
                    stack.pop_back();
                }
            } // while (!stack.empty())
        }
    }
}