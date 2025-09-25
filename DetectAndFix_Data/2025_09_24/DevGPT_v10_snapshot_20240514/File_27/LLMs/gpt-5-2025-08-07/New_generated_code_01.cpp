#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>
#include <limits>
#include <algorithm>
#include <cstdint>

class TargetDummy {
public:
    explicit TargetDummy(int64_t initialHealth) {
        // Validate and sanitize initial health
        if (initialHealth < 0) {
            health_ = 0;
        } else {
            health_ = initialHealth;
        }
    }

    bool isAlive() const { return health_ > 0; }

    int64_t health() const { return health_; }

    // Saturating damage to avoid underflow/overflow; rejects negative damage.
    void takeDamage(uint64_t damage) {
        if (damage == 0 || health_ == 0) return;
        if (damage >= static_cast<uint64_t>(health_)) {
            health_ = 0;
        } else {
            health_ -= static_cast<int64_t>(damage);
        }
    }

private:
    int64_t health_{0};
};

class DamageOverTimeAbility {
public:
    // Store damage as integer to avoid UB from float->int conversions later.
    uint64_t damagePerTick{0};
    double cooldownSeconds{1.0};
    double durationSeconds{1.0};
    double lastCastTime{-1.0};
    double lastDamageTime{-1.0};
    bool isActive{false};

    DamageOverTimeAbility(double damage, double cooldown, double duration) {
        // Validate inputs (CWE-20). Sanitize NaN/inf and enforce minimums.
        damagePerTick = sanitizeDamage(damage);
        cooldownSeconds = sanitizePositiveTime(cooldown, 0.001);  // minimum cooldown
        durationSeconds = sanitizePositiveTime(duration, 0.001);  // minimum duration
        // Initialize times so that first tick after cast can deal damage immediately.
        lastCastTime = -durationSeconds;
        lastDamageTime = -durationSeconds;
        isActive = false;
    }

    bool canDealDamage(double elapsedTime) const {
        if (!isActive) return false;
        // No epsilon needed if we ensure monotonic time and >= comparison
        // Guard against NaN (elapsedTime should be finite in this program).
        return (elapsedTime - lastDamageTime) >= cooldownSeconds;
    }

    void castAbility(double elapsedTime) {
        if (!isActive) {
            lastCastTime = elapsedTime;
            isActive = true;
            // Allow immediate damage on cast if cooldown already satisfied by design.
            // Keep lastDamageTime unchanged so first check uses cooldown vs. previous value.
        }
    }

    void updateActivity(double elapsedTime) {
        if (!isActive) return;
        if ((elapsedTime - lastCastTime) >= durationSeconds) {
            isActive = false;
        }
    }

private:
    static uint64_t sanitizeDamage(double d) {
        if (!std::isfinite(d) || d <= 0.0) return 0;
        // Clamp to uint64_t range; llround for better mapping, then clamp.
        long double ld = static_cast<long double>(d);
        if (ld > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            return std::numeric_limits<uint64_t>::max();
        }
        long long rounded = std::llround(ld);
        if (rounded <= 0) return 0;
        return static_cast<uint64_t>(rounded);
    }

    static double sanitizePositiveTime(double t, double minVal) {
        if (!std::isfinite(t) || t < minVal) return minVal;
        return t;
    }
};

int main() {
    // Create a target dummy with an initial health value (sanitize inside class)
    TargetDummy dummy(100);

    // Sanitize ability parameters in constructor
    DamageOverTimeAbility ability1(15.0, 10.0, 30.0);
    DamageOverTimeAbility ability2(10.0, 5.0, 20.0);

    int ticksPerSecond = 10; // Can be config; validate below

    // CWE-369: Division by zero defense and upper bound to avoid abusive CPU rates
    if (ticksPerSecond <= 0) {
        std::cerr << "Invalid ticksPerSecond (<=0). Clamping to 1.\n";
        ticksPerSecond = 1;
    } else if (ticksPerSecond > 1000) {
        std::cerr << "Excessive ticksPerSecond. Clamping to 1000 to avoid CPU abuse.\n";
        ticksPerSecond = 1000;
    }

    uint64_t ticks = 0;
    double activationCooldown = 1.0; // Shared activation cooldown

    using clock = std::chrono::steady_clock;
    const auto tickPeriod = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / static_cast<double>(ticksPerSecond)));
    auto start = clock::now();
    auto lastTickTime = start;
    auto nextTickTime = start;

    while (dummy.isAlive()) {
        auto now = clock::now();
        double elapsedTimeSeconds = std::chrono::duration<double>(now - start).count();
        double dtSeconds = std::chrono::duration<double>(now - lastTickTime).count();
        lastTickTime = now;

        // Update ability activity status first (prevents dealing damage after expiration)
        ability1.updateActivity(elapsedTimeSeconds);
        ability2.updateActivity(elapsedTimeSeconds);

        // If the shared activation cooldown is over, and no ability is active, cast the next ability
        if (activationCooldown <= 0.0 && !ability1.isActive && !ability2.isActive) {
            // Deterministic: cast ability1 then ability2
            if (!ability1.isActive) {
                ability1.castAbility(elapsedTimeSeconds);
            } else if (!ability2.isActive) {
                ability2.castAbility(elapsedTimeSeconds);
            }
            activationCooldown = 1.0; // Reset shared activation cooldown
        }

        // Compute canDealDamage once per ability per tick (avoids TOCTOU-like pattern)
        bool a1DamageNow = ability1.canDealDamage(elapsedTimeSeconds);
        bool a2DamageNow = ability2.canDealDamage(elapsedTimeSeconds);

        if (a1DamageNow) {
            std::cout << "Ability 1 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)\n";
            dummy.takeDamage(ability1.damagePerTick);
            ability1.lastDamageTime = elapsedTimeSeconds;
        }

        if (a2DamageNow) {
            std::cout << "Ability 2 used at Tick " << ticks << " (Time: " << elapsedTimeSeconds << "s)\n";
            dummy.takeDamage(ability2.damagePerTick);
            ability2.lastDamageTime = elapsedTimeSeconds;
        }

        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health() << "\n";

        if (!dummy.isAlive()) break;

        // Update the shared activation cooldown based on real elapsed time
        if (activationCooldown > 0.0) {
            activationCooldown = std::max(0.0, activationCooldown - dtSeconds);
        }

        // CWE-400: enforce tick pacing to avoid pegging CPU
        nextTickTime += tickPeriod;
        std::this_thread::sleep_until(nextTickTime);

        ++ticks;
    }

    double totalElapsedTimeSeconds =
        std::chrono::duration<double>(clock::now() - start).count();
    std::cout << "Target dummy killed in " << totalElapsedTimeSeconds << " seconds.\n";

    return 0;
}