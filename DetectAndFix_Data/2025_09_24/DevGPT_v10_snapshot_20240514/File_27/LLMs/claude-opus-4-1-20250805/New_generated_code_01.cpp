#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdint>

// Define a target dummy with health
class TargetDummy {
private:
    int32_t health;
    static constexpr int32_t MAX_HEALTH = 1000000;

public:
    TargetDummy(int32_t initialHealth) : health(initialHealth) {
        if (initialHealth <= 0 || initialHealth > MAX_HEALTH) {
            throw std::invalid_argument("Initial health must be between 1 and " + 
                                       std::to_string(MAX_HEALTH));
        }
    }
    
    bool isAlive() const { return health > 0; }
    
    int32_t getHealth() const { return health; }
    
    void takeDamage(int32_t damage) {
        if (damage < 0) {
            throw std::invalid_argument("Damage cannot be negative");
        }
        if (damage > MAX_HEALTH) {
            damage = MAX_HEALTH; // Cap damage to prevent overflow
        }
        health = (damage >= health) ? 0 : health - damage;
    }
};

class DamageOverTimeAbility {
private:
    static constexpr double EPSILON = 1e-9;
    static constexpr double MAX_DURATION = 3600.0; // 1 hour max
    static constexpr double MAX_DAMAGE = 10000.0;
    
public:
    double damagePerTick;
    double cooldownSeconds;
    double durationSeconds;
    double lastCastTime;
    double lastDamageTime;
    bool isActive;

    DamageOverTimeAbility(double damage, double cooldown, double duration)
        : damagePerTick(damage), cooldownSeconds(cooldown), durationSeconds(duration),
          lastCastTime(-duration), lastDamageTime(-duration), isActive(false) {
        
        // Input validation
        if (damage <= 0 || damage > MAX_DAMAGE) {
            throw std::invalid_argument("Damage must be between 0 and " + 
                                       std::to_string(MAX_DAMAGE));
        }
        if (cooldown <= 0 || cooldown > MAX_DURATION) {
            throw std::invalid_argument("Cooldown must be between 0 and " + 
                                       std::to_string(MAX_DURATION));
        }
        if (duration <= 0 || duration > MAX_DURATION) {
            throw std::invalid_argument("Duration must be between 0 and " + 
                                       std::to_string(MAX_DURATION));
        }
    }

    bool canDealDamage(double elapsedTime) const {
        return isActive && (elapsedTime - lastDamageTime >= cooldownSeconds - EPSILON);
    }

    void castAbility(double elapsedTime) {
        if (!isActive && elapsedTime >= 0) {
            lastCastTime = elapsedTime;
            isActive = true;
        }
    }

    void updateActivity(double elapsedTime) {
        if (isActive && (elapsedTime - lastCastTime >= durationSeconds - EPSILON)) {
            isActive = false;
        }
    }
    
    void updateLastDamageTime(double elapsedTime) {
        lastDamageTime = elapsedTime;
    }
};

int main() {
    try {
        // Create a target dummy with an initial health value
        TargetDummy dummy(100);

        DamageOverTimeAbility ability1(15.0, 10.0, 30.0);
        DamageOverTimeAbility ability2(10.0, 5.0, 20.0);

        const int32_t ticksPerSecond = 10;
        const int64_t MAX_TICKS = 100000; // Prevent infinite loops
        
        if (ticksPerSecond <= 0 || ticksPerSecond > 1000) {
            throw std::invalid_argument("Invalid ticks per second");
        }

        int64_t ticks = 0;
        double elapsedTimeSeconds = 0.0;
        double activationCooldown = 1.0;
        const double tickDuration = 1.0 / static_cast<double>(ticksPerSecond);

        while (dummy.isAlive() && ticks < MAX_TICKS) {
            // Check if ability 1 is active and can deal damage
            if (ability1.canDealDamage(elapsedTimeSeconds)) {
                std::cout << "Ability 1 used at Tick " << ticks 
                         << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
                
                // Safe damage conversion with bounds checking
                int32_t damage = static_cast<int32_t>(std::min(ability1.damagePerTick, 
                                                               static_cast<double>(std::numeric_limits<int32_t>::max())));
                dummy.takeDamage(damage);
                ability1.updateLastDamageTime(elapsedTimeSeconds);
            }

            // Check if ability 2 is active and can deal damage
            if (ability2.canDealDamage(elapsedTimeSeconds)) {
                std::cout << "Ability 2 used at Tick " << ticks 
                         << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
                
                int32_t damage = static_cast<int32_t>(std::min(ability2.damagePerTick, 
                                                               static_cast<double>(std::numeric_limits<int32_t>::max())));
                dummy.takeDamage(damage);
                ability2.updateLastDamageTime(elapsedTimeSeconds);
            }

            // Update ability activity status
            ability1.updateActivity(elapsedTimeSeconds);
            ability2.updateActivity(elapsedTimeSeconds);

            // Fixed logic: properly handle ability casting
            if (activationCooldown <= 0.0) {
                if (!ability1.isActive && !ability2.isActive) {
                    ability1.castAbility(elapsedTimeSeconds);
                    activationCooldown = 1.0;
                } else if (ability1.isActive && !ability2.isActive) {
                    ability2.castAbility(elapsedTimeSeconds);
                    activationCooldown = 1.0;
                }
            }

            // Display the dummy's current health
            std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.getHealth() << std::endl;

            // Update the shared activation cooldown
            if (activationCooldown > 0.0) {
                activationCooldown = std::max(0.0, activationCooldown - tickDuration);
            }
            
            elapsedTimeSeconds += tickDuration;
            ticks++;
        }

        if (ticks >= MAX_TICKS) {
            std::cerr << "Warning: Maximum tick limit reached. Simulation terminated." << std::endl;
        } else {
            double totalElapsedTimeSeconds = static_cast<double>(ticks) / static_cast<double>(ticksPerSecond);
            std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}