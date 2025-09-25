#include <unordered_map>
#include <list>
#include <stack>
#include <string>
#include <set>
#include <limits>
#include <stdexcept>

using namespace std;

// Assuming 'edge' is defined as:
struct edge {
    string station;
    double distance;
};

// Helper function to validate double values (e.g., not negative, not NaN, not infinite)
bool isValidDouble(double value) {
    return value >= 0.0 && value < std::numeric_limits<double>::infinity();
}

// Function definition for _88MPH, which finds a path through charging stations considering battery constraints
stack<string> _88MPH(
    const unordered_map<string, list<edge>>& mapOfChargingStations,
    const unordered_map<string, double>& chargingStation,
    const string& startingLocation,
    double initialBattery,
    set<string>& visited // For cycle detection
){
    stack<string> stationPath;

    // Input validation: Check if startingLocation exists in both maps
    if (mapOfChargingStations.find(startingLocation) == mapOfChargingStations.end() ||
        chargingStation.find(startingLocation) == chargingStation.end()) {
        // Invalid starting location
        return {};
    }

    // Prevent cycles (CWE-674)
    if (visited.count(startingLocation)) {
        return {};
    }
    visited.insert(startingLocation);

    double currentBattery = initialBattery;
    double charge = chargingStation.at(startingLocation);

    // Validate charge value (CWE-20)
    if (!isValidDouble(charge)) {
        return {};
    }

    // If the initial battery is zero or negative, return an empty path
    if(currentBattery <= 0.0)
        return {};

    // Charge the battery based on the charging station's power
    currentBattery = currentBattery + charge;

    // If the battery charge exceeds 75.5, return an empty path
    if(currentBattery > 75.5)
        return {};

    // If the battery charge is between 62.5 and 75.5 (inclusive), push the starting location onto the path and return
    if(currentBattery >= 62.5 && currentBattery <= 75.5){
        stationPath.push(startingLocation);
        return stationPath;
    }

    // Iterate over neighbors of the starting location in the map of charging stations
    for(const auto& neighbor : mapOfChargingStations.at(startingLocation)){
        // Validate neighbor's distance (CWE-20)
        if (!isValidDouble(neighbor.distance)) {
            continue;
        }
        // Calculate battery decrease based on distance traveled
        double decreaseBattery = neighbor.distance * 0.346;
        if (decreaseBattery > currentBattery) {
            continue; // Not enough battery to reach this neighbor
        }

        // Recursively call _88MPH function with updated parameters
        stack<string> resultPath = _88MPH(
            mapOfChargingStations,
            chargingStation,
            neighbor.station,
            currentBattery - decreaseBattery,
            visited
        );

        // If a non-empty path is found, push the starting location onto the path and break the loop
        if(!resultPath.empty()){
            resultPath.push(startingLocation);
            return resultPath;
        }
    }

    // Remove from visited set before returning (for other paths)
    visited.erase(startingLocation);

    // Return the resulting station path
    return {};
}

// Wrapper to hide 'visited' parameter from API users
stack<string> _88MPH(
    const unordered_map<string, list<edge>>& mapOfChargingStations,
    const unordered_map<string, double>& chargingStation,
    const string& startingLocation,
    double initialBattery
){
    set<string> visited;
    return _88MPH(mapOfChargingStations, chargingStation, startingLocation, initialBattery, visited);
}