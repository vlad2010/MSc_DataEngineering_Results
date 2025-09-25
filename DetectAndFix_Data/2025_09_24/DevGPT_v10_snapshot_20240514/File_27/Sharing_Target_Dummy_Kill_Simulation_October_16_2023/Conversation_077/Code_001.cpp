// Define an Ability class with additional name field
class Ability {
public:
    // ... (Previous members)

    std::string name; // Name or identifier for the ability

    Ability(double damage, double interval, double duration, double cooldown, int pri, const std::string& abilityName)
        : damagePerTick(damage), damageInterval(interval), durationSeconds(duration),
          lastCastTime(-cooldown), lastDamageTime(-interval), individualCooldown(cooldown), isActive(false), priority(pri), name(abilityName) {}

    // ... (Previous methods)
};

// ... (Previous code)

while (dummy.isAlive()) {
    for (Ability& ability : abilities) {
        // Update ability activity status
        ability.updateActivity(elapsedTimeSeconds);

        // If the ability is inactive and the shared cooldown has passed and individual cooldown is over, cast it
        if (ability.canCast(elapsedTimeSeconds, sharedCooldown)) {
            ability.castAbility(elapsedTimeSeconds);
            sharedCooldown = 1.0; // Set shared cooldown to 1 second after casting
        }

        // Check if the ability is active and can deal damage
        if (ability.canDealDamage(elapsedTimeSeconds)) {
            // Deal damage to the dummy
            dummy.takeDamage(static_cast<int>(ability.damagePerTick));
            // Update the last damage time
            ability.lastDamageTime = elapsedTimeSeconds;

            // Print the ability name, current tick, time, and dummy's health after taking damage
            std::cout << "Ability: " << ability.name << " | Tick: " << ticks << " | Time: " << elapsedTimeSeconds
                      << "s | Dummy Health: " << dummy.health << std::endl;
        }
    }

    // Update shared cooldown
    if (sharedCooldown > 0.0) {
        sharedCooldown -= 1.0 / ticksPerSecond;
    }

    // Simulate game logic for the rest of the tick
    // ...

    // Manually control the tick rate
    double tickDuration = 1.0 / ticksPerSecond;
    elapsedTimeSeconds += tickDuration;

    ticks++;
}

// ... (Remaining code)