#include <iostream>
#include <limits>
#include <cmath>

// Safe clamp helpers
namespace safe {
    // Clamp double to a non-negative int, handling NaN/Inf and range issues.
    inline int clampNonNegativeDoubleToInt(double v) {
        if (!std::isfinite(v) || v <= 0.0) {
            return 0;
        }
        const double maxInt = static_cast<double>(std::numeric_limits<int>::max());
        if (v >= maxInt) {
            return std::numeric_limits<int>::max();
        }
        // Round to nearest; use floor if you prefer truncation.
        return static_cast<int>(std::lround(v));
    }

    // Saturated subtraction for int: returns max(0, a - b)
    inline int sat_sub_nonneg(int a, int b) {
        if (b <= 0) return a;               // no change or healing not allowed here
        if (a <= b) return 0;
        return a - b;
    }

    // Validate double is finite and >= 0
    inline bool is_valid_nonneg_double(double v) {
        return std::isfinite(v) && v >= 0.0;
    }
}

// Define a target dummy with health
class TargetDummy {
public:
    explicit TargetDummy(int initialHealth) {
        if (initialHealth < 0) {
            health = 0;
        } else {
            health = initialHealth;
        }
    }

    bool isAlive() const { return health > 0; }

    // Only allow non-negative damage and saturate at 0
    void takeDamage(int damage) {
        if (damage < 0) {
            // Reject negative damage (which would effectively heal)
            return;
        }
        health = safe::sat_sub_nonneg(health, damage);
    }

    int getHealth() const { return health; }

private:
    int health;
};

class DamageOverTimeAbility {
public:
    DamageOverTimeAbility(double damage, double cooldown, double duration)
        : damagePerTick(damage),
          cooldownSeconds(cooldown),
          durationSeconds(duration),
          lastUsedTime(-cooldown) {}

    // Validate parameters to avoid NaN/Inf/negative issues
    bool isValid() const {
        return safe::is_valid_nonneg_double(damagePerTick) &&
               safe::is_valid_nonneg_double(cooldownSeconds) &&
               safe::is_valid_nonneg_double(durationSeconds);
    }

    bool canUse(double elapsedTime) const {
        return (elapsedTime - lastUsedTime) >= cooldownSeconds &&
               elapsedTime <= durationSeconds;
    }

    void use(double elapsedTime) {
        lastUsedTime = elapsedTime;
    }

    // Safely convert damage to int for application
    int damagePerTickInt() const {
        return safe::clampNonNegativeDoubleToInt(damagePerTick);
    }

    double getDuration() const { return durationSeconds; }

private:
    double damagePerTick;
    double cooldownSeconds;
    double durationSeconds;
    double lastUsedTime;
};

int main() {
    // Configuration (validate if coming from external input)
    const int ticksPerSecond = 10; // must be > 0
    if (ticksPerSecond <= 0 || ticksPerSecond > 100000) {
        std::cerr << "Invalid ticksPerSecond configuration." << std::endl;
        return 1;
    }

    // Create a target dummy with an initial health value
    TargetDummy dummy(100);

    DamageOverTimeAbility ability1(15.0, 10.0, 30.0); // 15 damage every 10s, up to 30s
    DamageOverTimeAbility ability2(10.0, 5.0, 20.0);  // 10 damage every 5s, up to 20s

    // Validate ability configs
    if (!ability1.isValid() || !ability2.isValid()) {
        std::cerr << "Invalid ability configuration detected." << std::endl;
        return 1;
    }

    int ticks = 0;
    double elapsedTimeSeconds = 0.0;
    const double tickDuration = 1.0 / static_cast<double>(ticksPerSecond);

    // Run while target is alive and at least one ability's window is open
    const double maxSimTime = std::max(ability1.getDuration(), ability2.getDuration());
    while (dummy.isAlive() && elapsedTimeSeconds <= maxSimTime) {
        if (ability1.canUse(elapsedTimeSeconds)) {
            dummy.takeDamage(ability1.damagePerTickInt());
            ability1.use(elapsedTimeSeconds);
            std::cout << "Ability 1 used at Tick " << ticks
                      << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        if (ability2.canUse(elapsedTimeSeconds)) {
            dummy.takeDamage(ability2.damagePerTickInt());
            ability2.use(elapsedTimeSeconds);
            std::cout << "Ability 2 used at Tick " << ticks
                      << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
        }

        ++ticks;

        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.getHealth() << std::endl;

        elapsedTimeSeconds += tickDuration;
    }

    const double totalElapsedTimeSeconds = static_cast<double>(ticks) / static_cast<double>(ticksPerSecond);

    if (dummy.isAlive()) {
        std::cout << "Target dummy survived " << totalElapsedTimeSeconds
                  << " seconds with " << dummy.getHealth() << " health." << std::endl;
    } else {
        std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds." << std::endl;
    }

    return 0;
}