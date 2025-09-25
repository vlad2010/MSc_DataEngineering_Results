#include <iostream>
#include <chrono>

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
    double abilityCooldownSeconds = 1.0; // Cooldown of the ability in seconds
    int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

    int ticks = 0;
    double abilityCooldown = 0.0;

    while (dummy.isAlive()) {
        // Check if the ability can be used (cooldown expired)
        if (abilityCooldown <= 0.0) {
            // Simulate using the ability
            dummy.takeDamage(static_cast<int>(abilityDamage));
            abilityCooldown = abilityCooldownSeconds;

            // Display ability usage
            std::cout << "Ability used at Tick " << ticks << std::endl;
        }

        ticks++;

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        abilityCooldown -= 1.0 / ticksPerSecond;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double elapsedTimeSeconds = static_cast<double>(ticks) / ticksPerSecond;
    std::cout << "Target dummy killed in " << elapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}