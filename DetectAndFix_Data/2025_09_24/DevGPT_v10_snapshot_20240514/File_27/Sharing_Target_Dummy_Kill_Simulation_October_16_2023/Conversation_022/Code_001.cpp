#include <iostream>
#include <vector>

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
    double cooldownSeconds;
    double durationSeconds;
    double lastUsedTime;
    bool isActive; // New property to indicate if the ability is active

    DamageOverTimeAbility(double damage, double cooldown, double duration)
        : damagePerTick(damage), cooldownSeconds(cooldown), durationSeconds(duration), lastUsedTime(-duration), isActive(false) {}

    bool canDealDamage(double elapsedTime) {
        return isActive && (elapsedTime - lastUsedTime) >= cooldownSeconds;
    }

    void castAbility(double elapsedTime) {
        lastUsedTime = elapsedTime;
        isActive = true;
    }

    void updateActivity(double elapsedTime) {
        if (elapsedTime - lastUsedTime >= durationSeconds) {
            isActive = false;
        }
    }
};

int main() {
    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    DamageOverTimeAbility ability1(15.0, 10.0, 30.0); // Damage ability: 15 damage every 10 seconds, for 30 seconds
    DamageOverTimeAbility ability2(10.0, 5.0, 20.0); // Another ability: 10 damage every 5 seconds, for 20 seconds

    int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;

    while (dummy.isAlive()) {
        // Check if ability 1 is active and can deal damage
        if (ability1.canDealDamage(elapsedTimeSeconds)) {
            // Simulate using the ability and dealing damage
            dummy.takeDamage(static_cast<int>(ability1.damagePerTick));

            // Display ability usage
            std::cout << "Ability 1 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        // Check if ability 2 is active and can deal damage
        if (ability2.canDealDamage(elapsedTimeSeconds)) {
            // Simulate using the ability and dealing damage
            dummy.takeDamage(static_cast<int>(ability2.damagePerTick));

            // Display ability usage
            std::cout << "Ability 2 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        // Update ability activity status
        ability1.updateActivity(elapsedTimeSeconds);
        ability2.updateActivity(elapsedTimeSeconds);

        // If an ability is inactive, cast it immediately
        if (!ability1.isActive) {
            ability1.castAbility(elapsedTimeSeconds);
        }
        if (!ability2.isActive) {
            ability2.castAbility(elapsedTimeSeconds);
        }

        ticks++;

        // Simulate game logic for the rest of the tick
        // ...

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        // Manually control the tick rate
        double tickDuration = 1.0 / ticksPerSecond;
        elapsedTimeSeconds += tickDuration;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double totalElapsedTimeSeconds = static_cast<double>(ticks) / ticksPerSecond;
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}