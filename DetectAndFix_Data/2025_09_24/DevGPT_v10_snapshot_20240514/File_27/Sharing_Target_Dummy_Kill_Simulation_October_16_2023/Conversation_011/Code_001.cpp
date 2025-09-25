// ...
int ticks = 0;
double abilityCooldown = 0.0;

while (dummy.isAlive()) {
    // Check if the ability can be used (cooldown expired)
    if (abilityCooldown <= 0.0) {
        // Simulate using the ability
        dummy.takeDamage(static_cast<int>(abilityDamage));
        abilityCooldown = abilityCooldownSeconds - (1.0 / ticksPerSecond); // Adjust for the current tick

        // Display ability usage
        std::cout << "Ability used at Tick " << ticks << std::endl;
    }

    ticks++;

    // Simulate game logic for the rest of the tick
    // ...

    // Display the dummy's current health
    std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;

    abilityCooldown -= 1.0 / ticksPerSecond;
}
// ...