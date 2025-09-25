#include <iostream>
#include <ctime>
#include <cstdlib>
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
    // Initialize random number generator
    srand(static_cast<unsigned int>(time(nullptr));

    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    double abilityDamage = 30.0; // Damage dealt by the ability
    double abilityCooldown = 1.0; // Cooldown of the ability in seconds

    int ticks = 0;
    double lastAbilityTime = 0.0; // Time when the ability was last used

    while (dummy.isAlive()) {
        // Calculate the elapsed time in seconds
        double currentTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;

        // Check if the ability can be used (cooldown expired)
        if (currentTime - lastAbilityTime >= abilityCooldown) {
            // Simulate using the ability
            dummy.takeDamage(static_cast<int>(abilityDamage));
            lastAbilityTime = currentTime;

            // Display ability usage
            std::cout << "Ability used at Tick " << ticks << " (Time: " << currentTime << "s)" << std::endl;
        }

        ticks++;

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        // Simulate an in-game tick (0.1 seconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double elapsedTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;
    std::cout << "Target dummy killed in " << elapsedTime << " seconds." << std::endl;

    return 0;
}