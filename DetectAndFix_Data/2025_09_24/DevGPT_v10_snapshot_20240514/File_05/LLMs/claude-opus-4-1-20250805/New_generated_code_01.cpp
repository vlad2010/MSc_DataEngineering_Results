#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stack>
#include <limits>
#include <filesystem>
#include <regex>

// Assuming vertex.hpp contains a template class with begin(), end(), and addEdge() methods
template<typename T>
class vertex {
    std::unordered_set<T> edges;
public:
    void addEdge(const T& edge) { edges.insert(edge); }
    auto begin() { return edges.begin(); }
    auto end() { return edges.end(); }
    bool empty() const { return edges.empty(); }
    size_t size() const { return edges.size(); }
};

// Constants for security limits
constexpr size_t MAX_ITERATIONS = 10000;
constexpr size_t MAX_RECURSION_DEPTH = 1000;
constexpr size_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
constexpr size_t MAX_LINE_LENGTH = 1024;
constexpr size_t MAX_NODES = 10000;

// Function prototypes
bool DFS(std::unordered_map<std::string, vertex<std::string>>& map,
         std::unordered_map<std::string, std::vector<std::string>>& keys);

std::string crossEdgesIterative(const std::string& current,
                                std::unordered_map<std::string, vertex<std::string>>& map,
                                std::unordered_map<std::string, std::vector<std::string>>& keys,
                                std::unordered_set<std::string>& visited);

bool isValidFileName(const std::string& fileName);
bool validateFileSize(const std::string& fileName);
bool validateLineFormat(const std::string& line, size_t expectedTokens);

int main() {
    try {
        std::string fileName;
        
        // Load police station map with security checks
        std::ifstream policeStationFile;
        int attempts = 0;
        const int MAX_ATTEMPTS = 3;
        
        do {
            if (attempts >= MAX_ATTEMPTS) {
                std::cerr << "Maximum attempts exceeded.\n";
                return 1;
            }
            
            std::cout << "Enter the police station file name: ";
            std::cin >> fileName;
            
            // Input validation
            if (!isValidFileName(fileName)) {
                std::cerr << "Invalid file name. Only alphanumeric characters, dots, and underscores allowed.\n";
                attempts++;
                continue;
            }
            
            if (!validateFileSize(fileName)) {
                std::cerr << "File is too large or doesn't exist.\n";
                attempts++;
                continue;
            }
            
            policeStationFile.open(fileName);
            attempts++;
        } while (!policeStationFile.is_open());
        
        // Construct the graph with validation
        std::unordered_map<std::string, vertex<std::string>> map;
        std::string line;
        size_t lineCount = 0;
        
        while (std::getline(policeStationFile, line)) {
            if (++lineCount > MAX_NODES) {
                std::cerr << "Too many nodes in the graph.\n";
                return 1;
            }
            
            if (line.length() > MAX_LINE_LENGTH) {
                std::cerr << "Line too long at line " << lineCount << "\n";
                return 1;
            }
            
            if (!validateLineFormat(line, 2)) {
                std::cerr << "Invalid format at line " << lineCount << "\n";
                continue;
            }
            
            std::istringstream iss(line);
            std::string key, value;
            if (iss >> key >> value) {
                // Validate string content
                if (key.empty() || value.empty() || 
                    key.length() > 100 || value.length() > 100) {
                    std::cerr << "Invalid data at line " << lineCount << "\n";
                    continue;
                }
                
                map[key].addEdge(value);
                map[value].addEdge(key);
            }
        }
        policeStationFile.close();
        
        // Get the key locations with validation
        std::unordered_map<std::string, std::vector<std::string>> keys;
        attempts = 0;
        
        do {
            if (attempts >= MAX_ATTEMPTS) {
                std::cerr << "Maximum attempts exceeded.\n";
                return 1;
            }
            
            std::cout << "Enter the keys file name: ";
            std::cin >> fileName;
            
            if (!isValidFileName(fileName)) {
                std::cerr << "Invalid file name.\n";
                attempts++;
                continue;
            }
            
            if (!validateFileSize(fileName)) {
                std::cerr << "File is too large or doesn't exist.\n";
                attempts++;
                continue;
            }
            
            policeStationFile.open(fileName);
            attempts++;
        } while (!policeStationFile.is_open());
        
        lineCount = 0;
        while (std::getline(policeStationFile, line)) {
            if (++lineCount > MAX_NODES) {
                std::cerr << "Too many key entries.\n";
                return 1;
            }
            
            if (line.length() > MAX_LINE_LENGTH) {
                std::cerr << "Line too long at line " << lineCount << "\n";
                return 1;
            }
            
            if (!validateLineFormat(line, 3)) {
                std::cerr << "Invalid format at line " << lineCount << "\n";
                continue;
            }
            
            std::istringstream iss(line);
            std::string key, value1, value2;
            if (iss >> key >> value1 >> value2) {
                if (key.empty() || value1.empty() || value2.empty() ||
                    key.length() > 100 || value1.length() > 100 || value2.length() > 100) {
                    std::cerr << "Invalid data at line " << lineCount << "\n";
                    continue;
                }
                
                keys[key].emplace_back(value1);
                keys[key].emplace_back(value2);
            }
        }
        policeStationFile.close();
        
        // Validate starting point exists
        if (map.find("MainHall") == map.end()) {
            std::cerr << "Starting point 'MainHall' not found in map.\n";
            return 1;
        }
        
        // Initiate graph traversal using Depth First Search
        bool success = DFS(map, keys);
        
        if (success) {
            std::cout << "Congratulations! You escaped the police station!\n";
        } else {
            std::cout << "Unfortunately, you were unable to escape the police station.\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

bool DFS(std::unordered_map<std::string, vertex<std::string>>& map,
         std::unordered_map<std::string, std::vector<std::string>>& keys) {
    std::unordered_set<std::string> visited;
    std::string room = "MainHall";
    visited.insert(room);
    std::string current;
    
    // Check if MainHall has edges
    if (map[room].empty()) {
        return false;
    }
    
    auto currentIt = map[room].begin();
    current = *currentIt;
    
    size_t iterations = 0;
    while (current != "Exit" && iterations < MAX_ITERATIONS) {
        iterations++;
        
        room = crossEdgesIterative(current, map, keys, visited);
        if (room == "End")
            break;
            
        if (!keys[room].empty() && keys[room].size() >= 2) {
            // Validate key room data
            const std::string& room1 = keys[room][0];
            const std::string& room2 = keys[room][1];
            
            if (map.find(room1) != map.end() && map.find(room2) != map.end()) {
                map[room1].addEdge(room2);
                map[room2].addEdge(room1);
                
                if (!map[room1].empty()) {
                    currentIt = map[room1].begin();
                } else {
                    break;
                }
            }
            keys[room].clear();
        } else {
            if (!map[room].empty()) {
                currentIt = map[room].begin();
            } else {
                break;
            }
        }
        
        current = *currentIt;
    }
    
    return current == "Exit";
}

std::string crossEdgesIterative(const std::string& start,
                                std::unordered_map<std::string, vertex<std::string>>& map,
                                std::unordered_map<std::string, std::vector<std::string>>& keys,
                                std::unordered_set<std::string>& visited) {
    // Use iterative approach to avoid stack overflow
    std::stack<std::string> stack;
    stack.push(start);
    size_t depth = 0;
    
    while (!stack.empty() && depth < MAX_RECURSION_DEPTH) {
        depth++;
        std::string current = stack.top();
        stack.pop();
        
        if (visited.find(current) != visited.end()) {
            continue;
        }
        
        visited.insert(current);
        
        if (!keys[current].empty()) {
            return current;
        }
        
        // Check if current exists in map
        if (map.find(current) != map.end()) {
            for (auto& neighbor : map[current]) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push(neighbor);
                }
            }
        }
    }
    
    return "End";
}

bool isValidFileName(const std::string& fileName) {
    // Only allow alphanumeric, dots, underscores, and forward slashes
    std::regex validPattern("^[a-zA-Z0-9._/]+$");
    
    // Prevent path traversal
    if (fileName.find("..") != std::string::npos) {
        return false;
    }
    
    return std::regex_match(fileName, validPattern) && fileName.length() < 256;
}

bool validateFileSize(const std::string& fileName) {
    try {
        if (!std::filesystem::exists(fileName)) {
            return false;
        }
        
        auto fileSize = std::filesystem::file_size(fileName);
        return fileSize <= MAX_FILE_SIZE;
    } catch (...) {
        return false;
    }
}

bool validateLineFormat(const std::string& line, size_t expectedTokens) {
    std::istringstream iss(line);
    std::string token;
    size_t count = 0;
    
    while (iss >> token) {
        count++;
        if (count > expectedTokens) {
            return false;
        }
    }
    
    return count == expectedTokens;
}