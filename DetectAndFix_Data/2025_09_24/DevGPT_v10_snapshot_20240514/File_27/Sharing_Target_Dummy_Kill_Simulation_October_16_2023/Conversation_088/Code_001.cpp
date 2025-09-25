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
    double damage;
    double interval;
    double duration;
    double cooldown;
    int priority;
    std::string name;
    double lastCastTime;
    double lastDamageTime;
    bool isActive;

    Ability(double dmg, double intrvl, double dur, double cd, int pri, const std::string& n)
        : damage(dmg), interval(intrvl), duration(dur), cooldown(cd), priority(pri), name(n),
          lastCastTime(-cd), lastDamageTime(-cd), isActive(false) {}

    bool canDealDamage(double elapsedTime) {
        // Use an epsilon (tolerance) to check for almost equality
        const double epsilon = 1e-5; // Adjust the value as needed
        return isActive && std::abs(elapsedTime - lastDamageTime) >= cooldown - epsilon;
    }

    void castAbility(double elapsedTime) {
        lastCastTime = elapsedTime;
        isActive = true;
    }

    void updateActivity(double elapsedTime) {
        // Use an epsilon (tolerance) to check for almost equality
        const double epsilon = 1e-5; // Adjust the value as needed
        if (std::abs(elapsedTime - lastCastTime) >= duration - epsilon) {
            isActive = false;
        }
    }
};

// Comparison function for sorting abilities by priority
bool compareAbilities(const Ability& a, const Ability& b) {
    return a.priority > b.priority;
}

int main() {
    double sharedCooldownValue = 1.0; // Set the shared cooldown value here
    int ticksPerSecond = 10; // Set the ticks per second value here

    // Create a target dummy with an initial health value (increased to 250)
    TargetDummy dummy(250);

    // Create a vector of abilities
    std::vector<Ability> abilities;
    abilities.push_back(Ability(15.0, 10.0, 30.0, 10.0, 2, "Firestorm"));
    abilities.push_back(Ability(10.0, 5.0, 20.0, 5.0, 1, "Frostbite"));
    abilities.push_back(Ability(20.0, 15.0, 10.0, 15.0, 3, "Thunderstrike"));
    abilities.push_back(Ability(5.0, 0.0, 0.0, 0.0, 0, "Venom Strike")); // New ability with 5 damage, no interval, duration, or cooldown, and lowest priority

    // Sort abilities by priority, from highest to lowest
    std::sort(abilities.begin(), abilities.end(), compareAbilities);

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;
    double sharedCooldown = 0.0;

    while (dummy.isAlive()) {
        for (Ability& ability : abilities) {
            // Check if the ability is active and can deal damage
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                // Display ability usage
                std::cout << ability.name << " used at Tick " << ticks
                          << " (Time: " << elapsedTimeSeconds << "s) - Health: " << dummy.health << std::endl;
            }

            // Update ability activity status
            ability.updateActivity(elapsedTimeSeconds);

            // If the ability is inactive and the shared cooldown has passed, cast it
            const double epsilon = 1e-5; // Adjust the value as needed
            if (!ability.isActive && sharedCooldown <= epsilon) {
                ability.castAbility(elapsedTimeSeconds);
                sharedCooldown = sharedCooldownValue; // Set shared cooldown to the specified value after casting
            }

            // Deal damage to the dummy if the ability is active
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                dummy.takeDamage(static_cast<int>(ability.damage));
                // Update the last damage time
                ability.lastDamageTime = elapsedTimeSeconds;
            }
        }

        // Update shared cooldown
        if (sharedCooldown > 0.0) {
            sharedCooldown -= 1.0 / ticksPerSecond;
        }

        // Manually control the tick rate
        elapsedTimeSeconds += 1.0 / ticksPerSecond;
        ticks++;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double totalElapsedTimeSeconds = static_cast<double>(ticks - 1) / ticksPerSecond;
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}