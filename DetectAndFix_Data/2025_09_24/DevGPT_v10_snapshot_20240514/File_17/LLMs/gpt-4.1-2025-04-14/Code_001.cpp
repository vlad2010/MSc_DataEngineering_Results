// Function definition for _88MPH, which finds a path through charging stations considering battery constraints
stack<string> _88MPH(unordered_map<string, list<edge>> mapOfChargingStations, unordered_map<string, double> chargingStation, string startingLocation, double initialBattery){
    // Initialize an empty stack to store the path
    stack<string> stationPath;
    // Declare a string variable to store the name of the current station
    string stationName = "";

    // Initialize variables to store battery charge and other parameters
    double currentBattery = initialBattery;
    double decreaseBattery = 0.0;
    double currentDistance = 0.0;
    double charge = chargingStation[startingLocation];

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
    for(auto& neighbor : mapOfChargingStations[startingLocation]){
        // Extract neighbor's station name and distance
        stationName = neighbor.station;
        currentDistance = neighbor.distance;
        // Calculate battery decrease based on distance traveled
        decreaseBattery = currentDistance * 0.346;
        // Recursively call _88MPH function with updated parameters
        stationPath = _88MPH(mapOfChargingStations, chargingStation, stationName, currentBattery-decreaseBattery);
        
        // If a non-empty path is found, push the starting location onto the path and break the loop
        if(!stationPath.empty()){
            stationPath.push(startingLocation);
            break;
        }
    }

    // Return the resulting station path
    return stationPath;
}