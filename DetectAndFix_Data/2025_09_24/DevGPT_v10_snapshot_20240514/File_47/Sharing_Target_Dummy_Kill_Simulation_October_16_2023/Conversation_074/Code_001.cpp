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
            // Display ability usage
            std::cout << "Ability with Priority " << ability.priority << " used at Tick " << ticks
                      << " (Time: " << elapsedTimeSeconds << "s)" << std::endl;
            // Deal damage to the dummy
            dummy.takeDamage(static_cast<int>(ability.damagePerTick));
            // Update the last damage time
            ability.lastDamageTime = elapsedTimeSeconds;
        }
    }

    // Update shared cooldown
    if (sharedCooldown > 0.0) {
        sharedCooldown -= 1.0 / ticksPerSecond;
    }

    // Simulate game logic for the rest of the tick
    // ...

    // Display the dummy's current health
    std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

    // Manually control the tick rate
    double tickDuration = 1.0 / ticksPerSecond;
    elapsedTimeSeconds += tickDuration;

    ticks++;
}

// ... (Remaining code)