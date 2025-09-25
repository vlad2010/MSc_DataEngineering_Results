#include <iostream>
#include <chrono>
#include <thread>

// Define a target dummy with health
class TargetDummy {
public:
    int health;

    TargetDummy(int initialHealth) : health(initialHealth) {}
    bool isAlive() { return health > 0; }
    void takeDamage(int damage) { health -= damage; }
};

int main() {
    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    double abilityDamage = 30.0; // Damage dealt by the ability
    double abilityCooldown = 10.0; // Cooldown of the ability in ticks (1 tick = 0.1 seconds)
    
    int ticks = 0;
    int abilityTicksRemaining = 0;

    while (dummy.isAlive()) {
        // Check if the ability can be used (cooldown expired)
        if (abilityTicksRemaining <= 0) {
            // Simulate using the ability
            dummy.takeDamage(static_cast<int>(abilityDamage));
            abilityTicksRemaining = static_cast<int>(abilityCooldown / 0.1); // Convert seconds to ticks

            // Display ability usage
            std::cout << "Ability used at Tick " << ticks << std::endl;
        }

        ticks++;

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        abilityTicksRemaining--;
    }

    // Calculate and display the time taken to kill the dummy in terms of ticks
    std::cout << "Target dummy killed in " << ticks << " ticks." << std::endl;

    return 0;
}