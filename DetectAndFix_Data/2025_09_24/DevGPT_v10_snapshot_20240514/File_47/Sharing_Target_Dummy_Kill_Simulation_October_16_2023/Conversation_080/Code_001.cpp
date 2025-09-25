int main() {
    double sharedCooldownValue = 1.0; // Set the shared cooldown value here
    int ticksPerSecond = 10; // Set the ticks per second value here

    // ... (Previous code)

    while (dummy.isAlive()) {
        for (Ability& ability : abilities) {
            // Update ability activity status
            ability.updateActivity(elapsedTimeSeconds);

            // If the ability is inactive and the shared cooldown has passed and individual cooldown is over, cast it
            if (ability.canCast(elapsedTimeSeconds, sharedCooldown)) {
                ability.castAbility(elapsedTimeSeconds);
                sharedCooldown = sharedCooldownValue; // Set shared cooldown using the variable
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
}