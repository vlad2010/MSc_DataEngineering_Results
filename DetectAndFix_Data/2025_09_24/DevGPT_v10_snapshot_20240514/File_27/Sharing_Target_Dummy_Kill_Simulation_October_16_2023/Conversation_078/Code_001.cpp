// ... (Previous code)

std::vector<Ability> abilities;
abilities.push_back(Ability(15.0, 10.0, 30.0, 10.0, 2, "Dragon's Fury")); // Ability 1
abilities.push_back(Ability(10.0, 5.0, 20.0, 5.0, 1, "Thunderstrike"));    // Ability 2
abilities.push_back(Ability(20.0, 15.0, 10.0, 15.0, 3, "Shadow Strike"));   // Ability 3
// Add more abilities with cool names and priorities as needed

// Sort abilities by priority, from highest to lowest
std::sort(abilities.begin(), abilities.end(), compareAbilities);

// ... (Rest of the code)