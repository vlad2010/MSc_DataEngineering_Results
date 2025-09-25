#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <limits>
#include <algorithm>
#include "vertex.hpp" // Assuming vertex.hpp is in the same directory

// Function prototypes
bool DFS(std::unordered_map<std::string, vertex<std::string>>& map,
         std::unordered_map<std::string, std::vector<std::string>>& keys);

std::string crossEdges(const std::string& current,
                       std::unordered_map<std::string, vertex<std::string>>& map,
                       std::unordered_map<std::string, std::vector<std::string>>& keys,
                       std::unordered_map<std::string, bool>& visited);

// Helper function to validate file name (basic check)
bool isValidFileName(const std::string& fileName) {
    if (fileName.empty()) return false;
    if (fileName.find("..") != std::string::npos) return false;
    if (fileName.find('/') != std::string::npos || fileName.find('\\') != std::string::npos) return false;
    return true;
}

int main() {
    std::string fileName;
    int attempts = 0;
    const int maxAttempts = 5;

    // Load police station map
    std::ifstream policeStationFile;
    do {
        if (attempts++ >= maxAttempts) {
            std::cerr << "Too many invalid attempts. Exiting.\n";
            return 1;
        }
        std::cout << "Enter the police station file name: ";
        std::cin >> fileName;
        if (!isValidFileName(fileName)) {
            std::cerr << "Invalid file name. Try again.\n";
            continue;
        }
        policeStationFile.open(fileName);
        if (!policeStationFile.is_open()) {
            std::cerr << "Could not open file. Try again.\n";
        }
    } while (!policeStationFile.is_open());

    // Construct the graph
    std::unordered_map<std::string, vertex<std::string>> map;
    std::string line;
    while (std::getline(policeStationFile, line)) {
        std::istringstream iss(line);
        std::string key, value;
        if (iss >> key >> value) {
            map[key].addEdge(value);
            map[value].addEdge(key);
        } else if (!line.empty()) {
            std::cerr << "Malformed line in police station file: " << line << "\n";
        }
    }
    if (policeStationFile.bad()) {
        std::cerr << "Error reading police station file.\n";
        return 1;
    }
    policeStationFile.close();

    // Get the key locations
    attempts = 0;
    do {
        if (attempts++ >= maxAttempts) {
            std::cerr << "Too many invalid attempts. Exiting.\n";
            return 1;
        }
        std::cout << "Enter the keys file name: ";
        std::cin >> fileName;
        if (!isValidFileName(fileName)) {
            std::cerr << "Invalid file name. Try again.\n";
            continue;
        }
        policeStationFile.open(fileName);
        if (!policeStationFile.is_open()) {
            std::cerr << "Could not open file. Try again.\n";
        }
    } while (!policeStationFile.is_open());

    std::unordered_map<std::string, std::vector<std::string>> keys;
    while (std::getline(policeStationFile, line)) {
        std::istringstream iss(line);
        std::string key, value1, value2;
        if (iss >> key >> value1 >> value2) {
            keys[key].emplace_back(value1);
            keys[key].emplace_back(value2);
        } else if (!line.empty()) {
            std::cerr << "Malformed line in keys file: " << line << "\n";
        }
    }
    if (policeStationFile.bad()) {
        std::cerr << "Error reading keys file.\n";
        return 1;
    }
    policeStationFile.close();

    // Initiate graph traversal using Depth First Search
    bool success = DFS(map, keys);

    if (success) {
        std::cout << "Congratulations! You escaped the police station!\n";
    } else {
        std::cout << "Unfortunately, you were unable to escape the police station.\n";
    }

    return 0;
}

bool DFS(std::unordered_map<std::string, vertex<std::string>>& map,
         std::unordered_map<std::string, std::vector<std::string>>& keys) {
    std::unordered_map<std::string, bool> visited;
    std::string room = "MainHall";
    visited[room] = true;
    std::string current;

    // Check if room exists in map
    auto mapIt = map.find(room);
    if (mapIt == map.end() || mapIt->second.begin() == mapIt->second.end()) {
        std::cerr << "MainHall not found or has no edges.\n";
        return false;
    }

    auto currentIt = mapIt->second.begin();
    current = *currentIt;

    while (current != "Exit") {
        room = crossEdges(current, map, keys, visited);
        if (room == "End")
            break;

        auto keysIt = keys.find(room);
        if (keysIt != keys.end() && !keysIt->second.empty()) {
            // Defensive: check if both keys exist
            if (keysIt->second.size() >= 2) {
                map[keysIt->second[0]].addEdge(keysIt->second[1]);
                map[keysIt->second[1]].addEdge(keysIt->second[0]);
                auto nextIt = map.find(keysIt->second[0]);
                if (nextIt == map.end() || nextIt->second.begin() == nextIt->second.end()) {
                    std::cerr << "Key room not found or has no edges.\n";
                    return false;
                }
                currentIt = nextIt->second.begin();
            }
            keysIt->second.clear();
        } else {
            auto nextIt = map.find(room);
            if (nextIt == map.end() || nextIt->second.begin() == nextIt->second.end()) {
                std::cerr << "Room not found or has no edges.\n";
                return false;
            }
            currentIt = nextIt->second.begin();
        }
        current = *currentIt;
    }

    return current == "Exit";
}

std::string crossEdges(const std::string& current,
                       std::unordered_map<std::string, vertex<std::string>>& map,
                       std::unordered_map<std::string, std::vector<std::string>>& keys,
                       std::unordered_map<std::string, bool>& visited) {
    visited[current] = true;

    auto keysIt = keys.find(current);
    if (keysIt != keys.end() && !keysIt->second.empty())
        return current;

    auto mapIt = map.find(current);
    if (mapIt == map.end()) return "End";

    for (auto& neighbor : mapIt->second) {
        if (!visited[neighbor]) {
            std::string result = crossEdges(neighbor, map, keys, visited);
            if (result != "End")
                return result;
        }
    }

    return "End";
}