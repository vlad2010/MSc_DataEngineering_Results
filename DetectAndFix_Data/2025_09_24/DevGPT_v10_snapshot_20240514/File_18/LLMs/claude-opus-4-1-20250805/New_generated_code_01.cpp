#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <limits>

class binTree {
public:
    struct binTreeNode {
        std::string data;
        std::shared_ptr<binTreeNode> left;
        std::shared_ptr<binTreeNode> right;
        
        explicit binTreeNode(const std::string& val) : data(val), left(nullptr), right(nullptr) {}
    };

private:
    static constexpr size_t MAX_TREE_DEPTH = 1000;  // Prevent stack overflow
    static constexpr size_t MAX_PATH_SIZE = 10000;  // Prevent memory exhaustion
    
    std::shared_ptr<binTreeNode> root;

public:
    // Secure zigzag traversal implementation
    std::vector<std::string> zigzag(const std::shared_ptr<binTreeNode>& r, 
                                    bool childType, 
                                    std::vector<std::string>& path) {
        // Input validation
        if (!r) {
            // Handle null pointer gracefully instead of crashing
            return path;  // Return current path unchanged
        }
        
        // Prevent memory exhaustion attacks
        if (path.size() >= MAX_PATH_SIZE) {
            throw std::runtime_error("Path size limit exceeded - possible malicious input");
        }
        
        // Use iterative approach with explicit depth tracking to prevent stack overflow
        return zigzagIterative(r, childType, path);
    }
    
private:
    // Iterative implementation to avoid stack overflow (CWE-674)
    std::vector<std::string> zigzagIterative(const std::shared_ptr<binTreeNode>& root,
                                             bool leftToRight,
                                             std::vector<std::string>& path) {
        if (!root) {
            return path;
        }
        
        std::vector<std::vector<std::shared_ptr<binTreeNode>>> levels;
        std::vector<std::shared_ptr<binTreeNode>> currentLevel;
        currentLevel.push_back(root);
        
        size_t depth = 0;
        
        while (!currentLevel.empty() && depth < MAX_TREE_DEPTH) {
            levels.push_back(currentLevel);
            std::vector<std::shared_ptr<binTreeNode>> nextLevel;
            
            for (const auto& node : currentLevel) {
                if (node) {  // Additional null check for safety
                    if (node->left) {
                        nextLevel.push_back(node->left);
                    }
                    if (node->right) {
                        nextLevel.push_back(node->right);
                    }
                }
            }
            
            currentLevel = std::move(nextLevel);
            depth++;
        }
        
        if (depth >= MAX_TREE_DEPTH) {
            throw std::runtime_error("Maximum tree depth exceeded - possible malicious input");
        }
        
        // Build zigzag path
        bool currentDirection = leftToRight;
        
        for (const auto& level : levels) {
            if (currentDirection) {
                // Left to right
                for (const auto& node : level) {
                    if (node && path.size() < MAX_PATH_SIZE) {
                        // Sanitize string data before adding to path
                        path.push_back(sanitizeString(node->data));
                    }
                }
            } else {
                // Right to left
                for (auto it = level.rbegin(); it != level.rend(); ++it) {
                    if (*it && path.size() < MAX_PATH_SIZE) {
                        // Sanitize string data before adding to path
                        path.push_back(sanitizeString((*it)->data));
                    }
                }
            }
            currentDirection = !currentDirection;
        }
        
        return path;
    }
    
    // String sanitization to prevent injection attacks
    std::string sanitizeString(const std::string& input) {
        // Limit string length to prevent memory issues
        constexpr size_t MAX_STRING_LENGTH = 1024;
        
        if (input.length() > MAX_STRING_LENGTH) {
            return input.substr(0, MAX_STRING_LENGTH);
        }
        
        // Additional sanitization could be added here based on requirements
        // For example, removing control characters, validating encoding, etc.
        std::string sanitized;
        sanitized.reserve(input.length());
        
        for (char c : input) {
            // Remove control characters except common ones like newline, tab
            if ((c >= 32 && c <= 126) || c == '\n' || c == '\t' || c == '\r') {
                sanitized += c;
            }
        }
        
        return sanitized;
    }
};