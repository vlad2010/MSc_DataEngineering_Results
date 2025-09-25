#include <iostream>
#include <vector>
#include <cmath> // For std::abs function

// Define a target dummy with health
class TargetDummy {
public:
    int health;

    TargetDummy(int initialHealth) : health(initialHealth) {}
    bool isAlive() { return health > 0; }
    void takeDamage(int damage) { health -= damage; }
};

class DamageOverTimeAbility {
public:
    double damagePerTick;
    double durationSeconds;
    double lastCastTime; // New property to track when the ability was last cast

    DamageOverTimeAbility(double damage, double duration)
        : damagePerTick(damage), durationSeconds(duration), lastCastTime(-duration) {}

    bool canDealDamage(double elapsedTime) {
        // Use an epsilon (tolerance) to check for almost equality
        const double epsilon = 1e-5; // Adjust the value as needed
        return std::abs(elapsedTime - lastCastTime) < durationSeconds - epsilon;
    }

    void castAbility(double elapsedTime) {
        lastCastTime = elapsedTime;
    }

    void updateActivity(double elapsedTime) {
        // No need to update activity in this version
    }
};

int main() {
    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    DamageOverTimeAbility ability1(15.0, 30.0); // Damage ability: 15 damage every 30 seconds
    DamageOverTimeAbility ability2(10.0, 20.0); // Another ability: 10 damage every 20 seconds

    int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;

    while (dummy.isAlive()) {
        // Check if ability 1 is active and can deal damage
        if (ability1.canDealDamage(elapsedTimeSeconds)) {
            // Display ability usage
            std::cout << "Ability 1 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        // Check if ability 2 is active and can deal damage
        if (ability2.canDealDamage(elapsedTimeSeconds)) {
            // Display ability usage
            std::cout << "Ability 2 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        // Deal damage to the dummy if abilities are active
        if (ability1.canDealDamage(elapsedTimeSeconds)) {
            dummy.takeDamage(static_cast<int>(ability1.damagePerTick));
        }
        if (ability2.canDealDamage(elapsedTimeSeconds)) {
            dummy.takeDamage(static_cast<int>(ability2.damagePerTick));
        }

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        // Control the tick rate
        double tickDuration = 1.0 / ticksPerSecond;
        elapsedTimeSeconds += tickDuration;

        ticks++;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double totalElapsedTimeSeconds = static_cast<double>(ticks) / ticksPerSecond;
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}