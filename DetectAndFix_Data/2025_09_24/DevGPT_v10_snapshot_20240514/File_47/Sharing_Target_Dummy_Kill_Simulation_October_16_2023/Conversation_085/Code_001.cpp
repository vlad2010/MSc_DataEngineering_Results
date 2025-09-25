int main() {
    double sharedCooldownValue = 1.0; // Set the shared cooldown value here
    int ticksPerSecond = 10; // Set the ticks per second value here

    // Create a target dummy with an initial health value (increased to 250)
    TargetDummy dummy(250);

    // Create a vector of abilities
    std::vector<Ability> abilities;
    abilities.push_back(Ability(15.0, 10.0, 30.0, 2, "Firestorm"));
    abilities.push_back(Ability(10.0, 5.0, 20.0, 1, "Frostbite"));
    abilities.push_back(Ability(20.0, 15.0, 10.0, 3, "Thunderstrike"));
    abilities.push_back(Ability(5.0, 0.0, 0.0, 0, "Venom Strike")); // New ability with 5 damage, no interval, duration, or cooldown, and lowest priority

    // Sort abilities by priority, from highest to lowest
    std::sort(abilities.begin(), abilities.end(), compareAbilities);

    // ... (Rest of the code)
}