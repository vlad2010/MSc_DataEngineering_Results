#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <cctype>

// Safe helper: returns true iff a and b differ by exactly one character.
// Prevents out-of-bounds by checking sizes first.
static bool oneCharDiff(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    int diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (++diff > 1) return false;
        }
    }
    return diff == 1;
}

// Optional: sanity check word content (letters only). Adjust if digits allowed.
static bool isAlphaWord(const std::string& s) {
    if (s.empty()) return false;
    for (unsigned char ch : s) {
        if (!std::isalpha(ch)) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Basic usage check (CWE-20)
    if (argc != 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "word_ladder") << " <wordbank-file>\n";
        return 1;
    }

    // Open file and validate (CWE-252)
    std::ifstream inFile(argv[1]);
    if (!inFile) {
        std::cerr << "Error: Unable to open file: " << argv[1] << "\n";
        return 1;
    }

    // Read dictionary safely (avoid while(!eof())) (CWE-20/CWE-252)
    std::vector<std::string> wordBank;
    std::string token;
    while (inFile >> token) {
        // Optionally filter out non-alpha entries to avoid surprises
        if (!token.empty() && isAlphaWord(token)) {
            wordBank.push_back(token);
        }
    }
    if (wordBank.empty()) {
        std::cerr << "Error: Word bank is empty or invalid.\n";
        return 1;
    }

    // Read start and target words, check extraction success (CWE-252)
    std::string start, target;
    if (!(std::cin >> start >> target)) {
        std::cerr << "Error: Failed to read start/target words from stdin.\n";
        return 1;
    }

    // Basic validation (CWE-20): non-empty, alphabetic, reasonable length
    // You may tailor max length based on constraints to mitigate DoS (CWE-400).
    constexpr size_t kMaxWordLen = 128; // Adjust as appropriate
    auto validWord = [&](const std::string& s) -> bool {
        return !s.empty() && s.size() <= kMaxWordLen && isAlphaWord(s);
    };
    if (!validWord(start) || !validWord(target)) {
        std::cerr << "Error: Invalid start/target word. Ensure alphabetic and length <= "
                  << kMaxWordLen << ".\n";
        return 1;
    }

    // The ladder requires same length words to compare safely (CWE-125)
    if (start.size() != target.size()) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    // Filter dictionary to words of the same length as start/target
    std::vector<std::string> bankSameLen;
    bankSameLen.reserve(wordBank.size());
    for (const auto& w : wordBank) {
        if (w.size() == start.size()) {
            bankSameLen.push_back(w);
        }
    }
    if (bankSameLen.empty()) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    // Use an unordered_set for fast membership checks
    std::unordered_set<std::string> dict(bankSameLen.begin(), bankSameLen.end());

    // If start/target not in dictionary, include them so BFS can traverse
    dict.insert(start);
    dict.insert(target);

    // BFS (iterative) to avoid uncontrolled recursion (CWE-674) and to be robust
    std::queue<std::string> q;
    std::unordered_set<std::string> visited;
    std::unordered_map<std::string, std::string> parent; // child -> parent

    q.push(start);
    visited.insert(start);

    bool found = false;
    while (!q.empty() && !found) {
        std::string cur = q.front();
        q.pop();

        // Generate neighbors by scanning dict (O(n)); for large dicts, consider
        // pattern-based adjacency to optimize.
        for (const auto& candidate : dict) {
            if (visited.find(candidate) != visited.end()) continue;
            if (oneCharDiff(cur, candidate)) {
                parent[candidate] = cur;
                if (candidate == target) {
                    found = true;
                    break;
                }
                visited.insert(candidate);
                q.push(candidate);
            }
        }
    }

    if (!found) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    // Reconstruct path from target to start
    std::vector<std::string> ladder;
    for (std::string at = target; !at.empty();) {
        ladder.push_back(at);
        auto it = parent.find(at);
        if (it == parent.end()) break;
        at = it->second;
    }
    // Ensure ladder ends with start; otherwise no valid path
    if (ladder.empty() || ladder.back() != start) {
        std::cout << "Solution Does Not Exist\n";
        return 0;
    }

    // Print ladder from start to target
    for (auto it = ladder.rbegin(); it != ladder.rend(); ++it) {
        std::cout << *it << "\n";
    }

    return 0;
}