#include <iostream>
#include <vector>
#include <cmath> // For std::abs function
#include <limits>

// Define a target dummy with health
class TargetDummy {
public:
    int health;

    TargetDummy(int initialHealth) : health(initialHealth) {
        // FIX: Input validation
        if (initialHealth <= 0 || initialHealth > 10000) {
            throw std::invalid_argument("Initial health must be between 1 and 10000.");
        }
    }
    bool isAlive() { return health > 0; }
    void takeDamage(int damage) {
        // FIX: Prevent integer underflow (CWE-190)
        if (damage < 0) return; // Ignore negative damage
        if (damage >= health) {
            health = 0;
        } else {
            health -= damage;
        }
    }
};

class DamageOverTimeAbility {
public:
    double damagePerTick;
    double cooldownSeconds;
    double durationSeconds;
    double lastCastTime;
    double lastDamageTime;
    bool isActive;

    DamageOverTimeAbility(double damage, double cooldown, double duration)
        : damagePerTick(damage), cooldownSeconds(cooldown), durationSeconds(duration),
          lastCastTime(0.0), lastDamageTime(0.0), isActive(false) {
        // FIX: Input validation
        if (damage <= 0.0 || cooldown <= 0.0 || duration <= 0.0) {
            throw std::invalid_argument("Ability parameters must be positive.");
        }
    }

    bool canDealDamage(double elapsedTime) {
        const double epsilon = 1e-5;
        return isActive && (elapsedTime - lastDamageTime) >= (cooldownSeconds - epsilon);
    }

    void castAbility(double elapsedTime) {
        if (!isActive) {
            lastCastTime = elapsedTime;
            lastDamageTime = elapsedTime - cooldownSeconds; // FIX: Ensure immediate damage on cast
            isActive = true;
        }
    }

    void updateActivity(double elapsedTime) {
        const double epsilon = 1e-5;
        if ((elapsedTime - lastCastTime) >= (durationSeconds - epsilon)) {
            isActive = false;
        }
    }
};

int main() {
    try {
        // Create a target dummy with an initial health value
        TargetDummy dummy(100); // You can set the initial health as per your requirements

        DamageOverTimeAbility ability1(15.0, 10.0, 30.0); // Damage ability: 15 damage every 10 seconds, for 30 seconds
        DamageOverTimeAbility ability2(10.0, 5.0, 20.0); // Another ability: 10 damage every 5 seconds, for 20 seconds

        int ticksPerSecond = 10; // Adjust this for the tick rate (e.g., 10 ticks per second)

        int ticks = 0;
        double elapsedTimeSeconds = 0.0;

        double activationCooldown = 1.0; // Shared activation cooldown

        // FIX: Add a maximum tick limit to prevent DoS (CWE-400)
        const int MAX_TICKS = 100000; // e.g., 10,000 seconds at 10 ticks/sec

        while (dummy.isAlive() && ticks < MAX_TICKS) {
            // Check if ability 1 is active and can deal damage
            if (ability1.canDealDamage(elapsedTimeSeconds)) {
                std::cout << "Ability 1 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            }

            // Check if ability 2 is active and can deal damage
            if (ability2.canDealDamage(elapsedTimeSeconds)) {
                std::cout << "Ability 2 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            }

            // Update ability activity status
            ability1.updateActivity(elapsedTimeSeconds);
            ability2.updateActivity(elapsedTimeSeconds);

            // If the shared activation cooldown is over, and no ability is active, cast the next ability
            if (activationCooldown <= 0.0 && !ability1.isActive && !ability2.isActive) {
                if (!ability1.isActive) {
                    ability1.castAbility(elapsedTimeSeconds);
                } else if (!ability2.isActive) {
                    ability2.castAbility(elapsedTimeSeconds);
                }
                activationCooldown = 1.0; // Reset shared activation cooldown
            }

            // Deal damage to the dummy if abilities are active
            if (ability1.canDealDamage(elapsedTimeSeconds)) {
                dummy.takeDamage(static_cast<int>(ability1.damagePerTick));
                ability1.lastDamageTime = elapsedTimeSeconds;
            }
            if (ability2.canDealDamage(elapsedTimeSeconds)) {
                dummy.takeDamage(static_cast<int>(ability2.damagePerTick));
                ability2.lastDamageTime = elapsedTimeSeconds;
            }

            // Simulate game logic for the rest of the tick
            // ...

            // Display the dummy's current health
            std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

            // Update the shared activation cooldown and control the tick rate
            if (activationCooldown > 0.0) {
                activationCooldown -= 1.0 / ticksPerSecond;
            }
            double tickDuration = 1.0 / ticksPerSecond;
            elapsedTimeSeconds += tickDuration;

            ticks++;
        }

        double totalElapsedTimeSeconds = static_cast<double>(ticks) / ticksPerSecond;
        if (!dummy.isAlive()) {
            std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;
        } else {
            std::cout << "Simulation stopped after reaching maximum ticks (" << MAX_TICKS << ")." << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}