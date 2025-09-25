#include <vector>
#include <string>
#include <stdexcept>

class binTreeNode {
public:
    std::string value;
    binTreeNode* left;
    binTreeNode* right;
    // ... other members ...
};

class binTree {
public:
    // Secure zigzag traversal
    std::vector<std::string> zigzag(const binTreeNode* r, bool childType, const std::vector<std::string>& path) const {
        // CWE-476: NULL Pointer Dereference - check input pointer
        if (r == nullptr) {
            throw std::invalid_argument("Input node pointer is null");
        }

        // Defensive copy of path to avoid modifying caller's data
        std::vector<std::string> result = path;

        // Example traversal logic (simplified)
        // Add current node's value
        result.push_back(r->value);

        // Traverse children based on childType
        if (childType) {
            if (r->left) {
                auto leftResult = zigzag(r->left, !childType, result);
                result.insert(result.end(), leftResult.begin(), leftResult.end());
            }
        } else {
            if (r->right) {
                auto rightResult = zigzag(r->right, !childType, result);
                result.insert(result.end(), rightResult.begin(), rightResult.end());
            }
        }

        // CWE-200: Information Exposure - ensure no sensitive data is returned
        // (Assume values are safe in this example)

        return result;
    }
};