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
    double abilityCooldownSeconds = 1.0; // Cooldown of the ability in seconds
    int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

    int ticks = 0;
    int abilityTicksRemaining = 0;

    while (dummy.isAlive()) {
        // Check if the ability can be used (cooldown expired)
        if (abilityTicksRemaining <= 0) {
            // Simulate using the ability
            dummy.takeDamage(static_cast<int>(abilityDamage));
            abilityTicksRemaining = static_cast<int>(abilityCooldownSeconds * ticksPerSecond);

            // Display ability usage
            std::cout << "Ability used at Tick " << ticks << std::endl;
        }

        ticks++;

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        abilityTicksRemaining--;
        
        // Simulate an in-game tick based on the tick rate
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / ticksPerSecond));
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double elapsedTimeSeconds = static_cast<double>(ticks) / ticksPerSecond;
    std::cout << "Target dummy killed in " << elapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}