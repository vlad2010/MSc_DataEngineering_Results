// ...

class Ability {
public:
    // ...

    bool canCast(double elapsedTime) {
        const double epsilon = 1e-5;
        return !isActive && (elapsedTime - lastCastTime >= cooldown - epsilon);
    }

    // ...
};

// ...

int main() {
    // ...

    while (dummy.isAlive()) {
        for (Ability& ability : abilities) {
            // Check if the ability can be cast
            if (ability.canCast(elapsedTimeSeconds)) {
                ability.castAbility(elapsedTimeSeconds);
                sharedCooldown = sharedCooldownValue; // Set shared cooldown after casting
            }

            // Check if the ability is active and can deal damage
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                // Display ability usage
                std::cout << ability.name << " used at Tick " << ticks
                          << " (Time: " << elapsedTimeSeconds << "s) - Health: " << dummy.health << std::endl;
            }

            // Update ability activity status
            ability.updateActivity(elapsedTimeSeconds);

            // Deal damage to the dummy if the ability is active
            if (ability.canDealDamage(elapsedTimeSeconds)) {
                dummy.takeDamage(static_cast<int>(ability.damage));
                // Update the last damage time
                ability.lastDamageTime = elapsedTimeSeconds;
            }
        }

        // ...
    }

    // ...

    return 0;
}