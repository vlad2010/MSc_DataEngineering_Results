#include <iostream>
#include <vector>
#include <ctime>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <random>

// Define a target dummy with health
class TargetDummy {
private:
    int health;
    static constexpr int MAX_HEALTH = 1000000; // Reasonable upper limit

public:
    TargetDummy(int initialHealth) {
        if (initialHealth < 0 || initialHealth > MAX_HEALTH) {
            throw std::invalid_argument("Initial health must be between 0 and " + 
                                       std::to_string(MAX_HEALTH));
        }
        health = initialHealth;
    }
    
    bool isAlive() const { return health > 0; }
    
    int getHealth() const { return health; }
    
    void takeDamage(int damage) {
        if (damage < 0) {
            throw std::invalid_argument("Damage cannot be negative");
        }
        
        // Prevent underflow - use saturating subtraction
        if (damage >= health) {
            health = 0;
        } else {
            health -= damage;
        }
    }
};

class DamageOverTimeAbility {
private:
    double damagePerTick;
    double cooldownSeconds;
    double durationSeconds;
    double lastUsedTime;
    static constexpr double EPSILON = 1e-9;
    static constexpr double MAX_DURATION = 3600.0; // 1 hour max
    static constexpr double MAX_DAMAGE = 10000.0;

public:
    DamageOverTimeAbility(double damage, double cooldown, double duration)
        : lastUsedTime(-cooldown) {
        
        // Input validation
        if (damage < 0 || damage > MAX_DAMAGE) {
            throw std::invalid_argument("Damage must be between 0 and " + 
                                       std::to_string(MAX_DAMAGE));
        }
        if (cooldown < 0 || cooldown > MAX_DURATION) {
            throw std::invalid_argument("Cooldown must be between 0 and " + 
                                       std::to_string(MAX_DURATION));
        }
        if (duration < 0 || duration > MAX_DURATION) {
            throw std::invalid_argument("Duration must be between 0 and " + 
                                       std::to_string(MAX_DURATION));
        }
        
        damagePerTick = damage;
        cooldownSeconds = cooldown;
        durationSeconds = duration;
    }

    bool canUse(double elapsedTime) const {
        // Use epsilon for floating point comparison
        return (elapsedTime - lastUsedTime >= cooldownSeconds - EPSILON) && 
               (elapsedTime <= durationSeconds + EPSILON);
    }

    void use(double elapsedTime) {
        if (elapsedTime < 0) {
            throw std::invalid_argument("Elapsed time cannot be negative");
        }
        lastUsedTime = elapsedTime;
    }
    
    double getDamagePerTick() const { return damagePerTick; }
    double getDurationSeconds() const { return durationSeconds; }
};

int main() {
    try {
        // Use better random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Create a target dummy with an initial health value
        TargetDummy dummy(100);

        DamageOverTimeAbility ability1(15.0, 10.0, 30.0);
        DamageOverTimeAbility ability2(10.0, 5.0, 20.0);

        int ticksPerSecond = 10;
        
        // Validate tick rate
        if (ticksPerSecond <= 0 || ticksPerSecond > 1000) {
            throw std::invalid_argument("Ticks per second must be between 1 and 1000");
        }

        // Use long long to prevent overflow for tick counter
        long long ticks = 0;
        const long long MAX_TICKS = 1000000; // Prevent infinite loops
        double elapsedTimeSeconds = 0.0;
        
        // Determine max duration from abilities
        double maxDuration = std::max(ability1.getDurationSeconds(), 
                                     ability2.getDurationSeconds());

        while (dummy.isAlive() && 
               elapsedTimeSeconds <= maxDuration && 
               ticks < MAX_TICKS) {
            
            // Check if ability 1 can be used
            if (ability1.canUse(elapsedTimeSeconds)) {
                // Safe conversion with bounds checking
                int damage = static_cast<int>(std::round(ability1.getDamagePerTick()));
                damage = std::min(damage, std::numeric_limits<int>::max());
                damage = std::max(damage, 0);
                
                dummy.takeDamage(damage);
                ability1.use(elapsedTimeSeconds);

                std::cout << "Ability 1 used at Tick " << ticks 
                         << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            }

            // Check if ability 2 can be used
            if (ability2.canUse(elapsedTimeSeconds)) {
                int damage = static_cast<int>(std::round(ability2.getDamagePerTick()));
                damage = std::min(damage, std::numeric_limits<int>::max());
                damage = std::max(damage, 0);
                
                dummy.takeDamage(damage);
                ability2.use(elapsedTimeSeconds);

                std::cout << "Ability 2 used at Tick " << ticks 
                         << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            }

            ticks++;

            // Display the dummy's current health
            std::cout << "Tick " << ticks << ": Dummy Health: " 
                     << dummy.getHealth() << std::endl;

            // Use precise tick duration calculation
            double tickDuration = 1.0 / static_cast<double>(ticksPerSecond);
            elapsedTimeSeconds = static_cast<double>(ticks) * tickDuration;
        }

        // Check if we hit the max tick limit
        if (ticks >= MAX_TICKS) {
            std::cerr << "Warning: Maximum tick limit reached" << std::endl;
        }

        // Calculate and display the time taken to kill the dummy
        if (!dummy.isAlive()) {
            double totalElapsedTimeSeconds = static_cast<double>(ticks) / 
                                            static_cast<double>(ticksPerSecond);
            std::cout << "Target dummy killed in " << totalElapsedTimeSeconds 
                     << " seconds." << std::endl;
        } else {
            std::cout << "Target dummy survived with " << dummy.getHealth() 
                     << " health." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}