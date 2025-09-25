#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <filesystem>
#include <cctype>
#include <limits>
#include "vertex.hpp" // Assuming vertex.hpp is in the same directory

// Security/robustness constraints
static constexpr std::size_t MAX_TOKEN_LEN = 64;
static constexpr std::size_t MAX_NODES     = 10000;
static constexpr std::size_t MAX_EDGES     = 50000;
static constexpr std::size_t MAX_KEYS      = 10000;

// Validate room/key token (alnum, '_', '-', length-limited)
bool isSafeToken(const std::string& s) {
    if (s.empty() || s.size() > MAX_TOKEN_LEN) return false;
    for (unsigned char c : s) {
        if (!(std::isalnum(c) || c == '_' || c == '-')) return false;
    }
    return true;
}

// Validate filename: no path separators, limited charset, length-limited
bool isSafeFilename(const std::string& name) {
    if (name.empty() || name.size() > MAX_TOKEN_LEN) return false;
    for (unsigned char c : name) {
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.')) return false;
    }
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return false;
    return true;
}

void trim(std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    auto b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) s.clear();
    else s = s.substr(a, b - a + 1);
}

bool promptAndOpenFile(std::ifstream& file, const std::string& prompt) {
    using fs = std::filesystem;
    for (int attempts = 0; attempts < 5; ++attempts) {
        std::cout << prompt;
        std::string input;
        if (!std::getline(std::cin, input)) {
            std::cerr << "Input error: unable to read file name." << std::endl;
            return false;
        }
        trim(input);
        if (!isSafeFilename(input)) {
            std::cerr << "Invalid file name. Use only [A-Za-z0-9_.-], no directories, length <= "
                      << MAX_TOKEN_LEN << "." << std::endl;
            continue;
        }

        fs::path p = fs::current_path() / input;
        std::error_code ec;
        auto st = fs::status(p, ec);
        if (ec || !fs::is_regular_file(st)) {
            std::cerr << "File not found or not a regular file: " << p.string() << std::endl;
            continue;
        }
        file.open(p, std::ios::in);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << p.string() << std::endl;
            continue;
        }
        return true;
    }
    std::cerr << "Too many invalid attempts. Aborting." << std::endl;
    return false;
}

// Iterative DFS that unlocks edges when keys are found
bool escapeDFS(std::unordered_map<std::string, vertex<std::string>>& graph,
               std::unordered_map<std::string, std::vector<std::string>>& keys) {
    const std::string start = "MainHall";
    const std::string goal  = "Exit";

    if (graph.find(start) == graph.end()) {
        std::cerr << "Start room '" << start << "' does not exist in the map." << std::endl;
        return false;
    }

    std::unordered_set<std::string> visited;
    std::vector<std::string> stack;
    stack.push_back(start);

    while (!stack.empty()) {
        std::string current = stack.back();
        stack.pop_back();

        if (!visited.insert(current).second) continue; // already visited
        if (current == goal) return true;

        // If current room has a key pair, unlock the edge (only if both nodes exist)
        auto kit = keys.find(current);
        if (kit != keys.end() && kit->second.size() >= 2) {
            const std::string& a = kit->second[0];
            const std::string& b = kit->second[1];

            // Only unlock edges between existing rooms to avoid unbounded growth
            auto ita = graph.find(a);
            auto itb = graph.find(b);
            if (ita != graph.end() && itb != graph.end()) {
                ita->second.addEdge(b);
                itb->second.addEdge(a);
            }
            // Clear to prevent repeated unlocking
            kit->second.clear();
        }

        // Explore neighbors safely
        auto mit = graph.find(current);
        if (mit != graph.end()) {
            // vertex<std::string> is assumed iterable
            for (const auto& neighbor : mit->second) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push_back(neighbor);
                }
            }
        }
    }

    return false;
}

int main() {
    // Load police station map (graph)
    std::ifstream policeStationFile;
    if (!promptAndOpenFile(policeStationFile, "Enter the police station file name: ")) {
        return 1;
    }

    std::unordered_map<std::string, vertex<std::string>> graph;
    std::string line;
    std::size_t edgeCount = 0;

    auto ensureNode = [&](const std::string& n) -> bool {
        if (graph.find(n) != graph.end()) return true;
        if (graph.size() >= MAX_NODES) {
            std::cerr << "Node limit reached (" << MAX_NODES << "). Skipping node: " << n << std::endl;
            return false;
        }
        graph.emplace(n, vertex<std::string>{});
        return true;
    };

    while (std::getline(policeStationFile, line)) {
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) {
            // Malformed line: skip
            continue;
        }
        if (!isSafeToken(a) || !isSafeToken(b)) {
            std::cerr << "Skipping invalid token(s) in graph: '" << a << "', '" << b << "'." << std::endl;
            continue;
        }
        if (edgeCount >= MAX_EDGES) {
            std::cerr << "Edge limit reached (" << MAX_EDGES << "). Remaining edges will be ignored." << std::endl;
            break;
        }
        if (!ensureNode(a) || !ensureNode(b)) continue;

        // Safe addEdge without creating unintended nodes
        auto ita = graph.find(a);
        auto itb = graph.find(b);
        ita->second.addEdge(b);
        itb->second.addEdge(a);
        ++edgeCount;
    }
    policeStationFile.close();

    // Load keys (unlock pairs)
    std::ifstream keysFile;
    if (!promptAndOpenFile(keysFile, "Enter the keys file name: ")) {
        return 1;
    }

    std::unordered_map<std::string, std::vector<std::string>> keys;
    std::size_t keyPairs = 0;

    while (std::getline(keysFile, line)) {
        if (keyPairs >= MAX_KEYS) {
            std::cerr << "Key pairs limit reached (" << MAX_KEYS << "). Remaining keys will be ignored." << std::endl;
            break;
        }
        std::istringstream iss(line);
        std::string k, v1, v2;
        if (!(iss >> k >> v1 >> v2)) {
            continue; // malformed
        }
        if (!isSafeToken(k) || !isSafeToken(v1) || !isSafeToken(v2)) {
            std::cerr << "Skipping invalid key line: '" << k << " " << v1 << " " << v2 << "'." << std::endl;
            continue;
        }
        // Store only the first pair for a room to avoid uncontrolled growth
        auto& vec = keys[k];
        if (vec.empty()) {
            vec.emplace_back(v1);
            vec.emplace_back(v2);
            ++keyPairs;
        }
    }
    keysFile.close();

    // Perform safe, iterative DFS to determine escape
    bool success = escapeDFS(graph, keys);

    if (success) {
        std::cout << "Congratulations! You escaped the police station!\n";
    } else {
        std::cout << "Unfortunately, you were unable to escape the police station.\n";
    }

    return 0;
}