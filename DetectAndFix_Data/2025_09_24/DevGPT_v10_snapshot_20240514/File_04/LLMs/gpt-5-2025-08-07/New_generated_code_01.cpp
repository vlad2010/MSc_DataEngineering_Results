#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>

// Compute Hamming distance safely; returns SIZE_MAX if lengths differ.
static size_t hammingDistance(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return static_cast<size_t>(-1);
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += (a[i] != b[i]);
    }
    return diff;
}

// Iterative BFS to find the shortest word ladder; avoids recursion (CWE-674) and
// ensures we never index out of bounds (CWE-125).
static std::vector<std::string> findWordLadder(
    const std::string& start,
    const std::string& target,
    const std::vector<std::string>& wordBank)
{
    std::vector<std::string> empty;

    if (start.size() == 0 || start.size() != target.size()) {
        return empty;
    }

    // Build a dictionary set filtered to the correct word length to avoid OOB comparisons.
    std::unordered_set<std::string> dict;
    dict.reserve(wordBank.size());
    for (const auto& w : wordBank) {
        if (w.size() == start.size()) {
            dict.insert(w);
        }
    }

    // If target not in dictionary, you may still want to allow it; include it explicitly.
    dict.insert(target);
    dict.insert(start);

    // BFS
    std::queue<std::string> q;
    std::unordered_set<std::string> visited;
    std::unordered_map<std::string, std::string> parent; // child -> parent

    q.push(start);
    visited.insert(start);
    parent[start] = std::string();

    while (!q.empty()) {
        std::string cur = q.front();
        q.pop();

        if (cur == target) {
            // Reconstruct path
            std::vector<std::string> path;
            for (std::string at = target; !at.empty(); at = parent[at]) {
                path.push_back(at);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Explore neighbors: words that differ by exactly one character.
        for (const auto& candidate : dict) {
            if (visited.find(candidate) != visited.end()) continue;
            size_t dist = hammingDistance(cur, candidate); // safe by length filter
            if (dist == 1) {
                visited.insert(candidate);
                parent[candidate] = cur;
                q.push(candidate);
            }
        }
    }

    // No ladder found
    return empty;
}

int main(int argc, char* argv[]) {
    // Validate command-line args (CWE-20, CWE-703)
    if (argc != 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "word_ladder") << " <wordbank_file>\n";
        return 1;
    }

    const char* filePath = argv[1];
    std::ifstream inFile(filePath);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file: " << filePath << "\n";
        return 2;
    }

    std::vector<std::string> wordBank;
    wordBank.reserve(10000); // Optional: pre-reserve

    // Safe tokenized read (avoid EOF anti-pattern) (fix CWE-703/CWE-252)
    std::string token;
    while (inFile >> token) {
        if (!token.empty()) {
            wordBank.push_back(token);
        }
    }
    if (inFile.bad()) {
        std::cerr << "Error: I/O error while reading " << filePath << "\n";
        return 3;
    }

    std::string start, target;
    if (!(std::cin >> start >> target)) {
        std::cerr << "Error: Failed to read start/target words from stdin\n";
        return 4;
    }

    // Validate inputs (CWE-20) and preconditions to avoid OOB (CWE-125)
    if (start.empty() || target.empty()) {
        std::cerr << "Error: Start/target words must be non-empty\n";
        return 5;
    }
    if (start.size() != target.size()) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    auto ladder = findWordLadder(start, target, wordBank);

    if (ladder.empty()) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    // Print ladder safely (we know ladder is non-empty here)
    for (const auto& w : ladder) {
        std::cout << w << '\n';
    }

    return 0;
}