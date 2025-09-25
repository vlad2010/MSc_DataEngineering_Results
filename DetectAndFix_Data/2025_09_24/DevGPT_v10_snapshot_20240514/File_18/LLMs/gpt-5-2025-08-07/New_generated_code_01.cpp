#include <string>
#include <vector>
#include <deque>

// Minimal node definition for demonstration.
// In real code, prefer documenting ownership clearly; smart pointers can be used
// for tree construction/ownership but traversal typically uses raw/observer pointers.
struct binTreeNode {
    std::string value;
    binTreeNode* left{nullptr};
    binTreeNode* right{nullptr};
};

class binTree {
public:
    // New, safer API: no recursion, no ambiguous flags, no out-parameters.
    static std::vector<std::string> zigzag(const binTreeNode* root) {
        std::vector<std::string> result;
        if (root == nullptr) {
            // CWE-476 mitigated: handle null safely.
            return result;
        }

        // Iterative zigzag using a deque to avoid recursion (mitigates CWE-674).
        std::deque<const binTreeNode*> dq;
        dq.push_back(root);
        bool leftToRight = true;

        while (!dq.empty()) {
            const std::size_t levelSize = dq.size();
            // Collect values for this level; we will append them to result.
            std::vector<std::string> level;
            level.reserve(levelSize);

            for (std::size_t i = 0; i < levelSize; ++i) {
                if (leftToRight) {
                    const binTreeNode* node = dq.front();
                    dq.pop_front();
                    level.push_back(node->value);
                    if (node->left)  dq.push_back(node->left);
                    if (node->right) dq.push_back(node->right);
                } else {
                    const binTreeNode* node = dq.back();
                    dq.pop_back();
                    level.push_back(node->value);
                    // When traversing from right to left, push children to the front
                    // in reverse order to maintain correct processing next iteration.
                    if (node->right) dq.push_front(node->right);
                    if (node->left)  dq.push_front(node->left);
                }
            }

            // Append this level's values to the overall result (flattened order).
            result.insert(result.end(), level.begin(), level.end());
            leftToRight = !leftToRight;
        }

        return result;
    }

    // Legacy API retained for compatibility but made safe.
    // - childType is ignored (ambiguous flag).
    // - path is ignored to avoid unexpected external mutations.
    // - Delegates to the safe, const-correct implementation above.
    [[deprecated("Use zigzag(const binTreeNode*) instead; this overload ignores childType and path")]]
    static std::vector<std::string> zigzag(binTreeNode* r, bool /*childType*/, std::vector<std::string>& /*path*/) {
        return zigzag(static_cast<const binTreeNode*>(r));
    }
};