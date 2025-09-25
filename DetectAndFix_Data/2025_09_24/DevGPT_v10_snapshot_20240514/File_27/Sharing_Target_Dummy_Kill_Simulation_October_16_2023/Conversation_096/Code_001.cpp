#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

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
    double damage;         // Damage per cast
    double damageInterval; // Interval between damage ticks
    double durationSeconds; // Duration of the ability
    double cooldown;       // Cooldown before the ability can be cast again
    int priority;          // Priority of the ability
    std::string name;      // Name of the ability
    double lastCastTime;   // Track when the ability was last cast
    double lastDamageTime; // Track when the ability last dealt damage
    bool isActive;         // Indicate if the ability is active

    Ability(double dmg, double interval, double duration, double cd, int pri, const std::string& n)
        : damage(dmg), damageInterval(interval), durationSeconds(duration), cooldown(cd), priority(pri), name(n),
          lastCastTime(-cd), lastDamageTime(-interval), isActive(false) {}
};

// Comparison function for sorting abilities by priority
bool compareAbilities(const Ability& a, const Ability& b) {
    return a.priority > b.priority;
}

int main() {
    // Create a target dummy with an initial health value
    TargetDummy dummy(250); // Updated initial health to 250

    std::vector<Ability> abilities;
    abilities.push_back(Ability(15.0, 10.0, 30.0, 10.0, 2, "Frostbolt")); // Ability 1: Frostbolt
    abilities.push_back(Ability(10.0, 5.0, 20.0, 5.0, 1, "Fireball"));    // Ability 2: Fireball
    abilities.push_back(Ability(20.0, 15.0, 10.0, 15.0, 3, "Thunderstrike")); // Ability 3: Thunderstrike
    abilities.push_back(Ability(5.0, 0.0, 0.0, 0.0, 0, "Basic Attack"));  // Ability 4: Basic Attack

    // Sort abilities by priority, from highest to lowest
    std::sort(abilities.begin(), abilities.end(), compareAbilities);

    double sharedCooldownValue = 1.0;  // Shared cooldown for ability casts
    int ticksPerSecond = 10;  // Ticks per second

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;
    double sharedCooldown = 0.0; // Shared cooldown for ability casts

    while (dummy.isAlive()) {
        for (Ability& ability : abilities) {
            // Update ability activity status
            ability.isActive = false;
            if (sharedCooldown <= 0.0) {
                if (elapsedTimeSeconds - ability.lastCastTime >= ability.cooldown) {
                    ability.isActive = true;
                    ability.lastCastTime = elapsedTimeSeconds;
                    sharedCooldown = sharedCooldownValue;
                }
            }

            // Check if the ability is active and can deal damage
            if (ability.isActive) {
                // Display ability usage
                std::cout << "Ability with Priority " << ability.priority << " (" << ability.name << ") used at Tick " << ticks
                          << " (Time: " << elapsedTimeSeconds << "s) - Damage: " << ability.damage << std::endl;

                // Deal damage to the dummy
                dummy.takeDamage(static_cast<int>(ability.damage));
                ability.lastDamageTime = elapsedTimeSeconds;
            }
        }

        // Update shared cooldown
        if (sharedCooldown > 0.0) {
            sharedCooldown -= 1.0 / ticksPerSecond;
        }

        // Display the dummy's current health when it takes damage
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

        elapsedTimeSeconds += 1.0 / ticksPerSecond;
        ticks++;
    }

    double totalElapsedTimeSeconds = static_cast<double>(ticks - 1) / ticksPerSecond;
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;

    return 0;
}