#include <iostream>
#include <vector>
#include <cmath> // For std::abs function
#include <algorithm> // For std::sort

// Define a target dummy with health
class TargetDummy {
public:
    int health;

    TargetDummy(int initialHealth) : health(initialHealth) {}
    bool isAlive() { return health > 0; }
    void takeDamage(int damage) { health -= damage; }
};

class Ability {
public:
    double damagePerTick;
    double damageInterval;
    double durationSeconds;
    double lastCastTime; // Track when the ability was last cast
    double lastDamageTime; // Track when the ability last dealt damage
    bool isActive; // Indicate if the ability is active
    int priority; // Priority of the ability

    Ability(double damage, double interval, double duration, int pri)
        : damagePerTick(damage), damageInterval(interval), durationSeconds(duration),
          lastCastTime(-duration), lastDamageTime(-interval), isActive(false), priority(pri) {}

    bool canDealDamage(double elapsedTime) {
        // Use an epsilon (tolerance) to check for almost equality
        const double epsilon = 1e-5; // Adjust the value as needed
        return isActive && std::abs(elapsedTime - lastDamageTime) >= damageInterval - epsilon;
    }

    void castAbility(double elapsedTime) {
        lastCastTime = elapsedTime;
        isActive = true;
    }

    void updateActivity(double elapsedTime) {
        // Use an epsilon (tolerance) to check for almost equality
        const double epsilon = 1e-5; // Adjust the value as needed
        if (std::abs(elapsedTime - lastCastTime) >= durationSeconds - epsilon) {
            isActive = false;
        }
    }
};

// Comparison function for sorting abilities by priority
bool compareAbilities(const Ability& a, const Ability& b) {
    return a.priority > b.priority;
}

int main() {
    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    std::vector<Ability> abilities;
    abilities.push_back(Ability(15.0, 10.0, 30.0, 2)); // Ability 1
    abilities.push_back(Ability(10.0, 5.0, 20.0, 1));  // Ability 2
    abilities.push_back(Ability(20.0, 15.0, 10.0, 3)); // Ability 3
    // Add more abilities with priorities as needed

    // Sort abilities by priority, from highest to lowest
    std::sort(abilities.begin(), abilities.end(), compareAbilities);

    int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;
    double sharedCooldown = 0.0; // Shared cooldown for ability casts

    while (dummy.isAlive()) {
        for (Ability& ability : abilities) {
            // Check if the ability is active and can deal damage
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                // Display ability usage
                std::cout << "Ability with Priority " << ability.priority << " used at Tick " << ticks
                          << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            }

            // Update ability activity status
            ability.updateActivity(elapsedTimeSeconds);

            // If the ability is inactive and the shared cooldown has passed, cast it
            const double epsilon = 1e-5; // Adjust the value as needed
            if (!ability.isActive && sharedCooldown <= epsilon) {
                ability.castAbility(elapsedTimeSeconds);
                sharedCooldown = 1.0; // Set shared cooldown to 1 second after casting
            }

            // Deal damage to the dummy if the ability is active
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                dummy.takeDamage(static_cast<int>(ability.damagePerTick));
                // Update the last damage time
                ability.lastDamageTime = elapsedTimeSeconds;
            }
        }

        // Update shared cooldown
        if (sharedCooldown > 0.0) {
            sharedCooldown -= 1.0 / ticksPerSecond;
        }

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        // Manually control the tick rate
        double tickDuration = 1.0 / ticksPerSecond;
        elapsedTimeSeconds += tickDuration;

        ticks++;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double totalElapsedTimeSeconds = static_cast<double>(ticks - 1) / ticksPerSecond;
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}