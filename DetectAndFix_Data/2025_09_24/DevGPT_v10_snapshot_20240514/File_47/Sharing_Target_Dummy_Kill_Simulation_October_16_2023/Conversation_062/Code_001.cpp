class DamageOverTimeAbility {
public:
    int damage;           // Single instance damage
    double cooldownSeconds;
    double lastCastTime;  // Track when the ability was last cast

    DamageOverTimeAbility(int damage, double cooldown)
        : damage(damage), cooldownSeconds(cooldown), lastCastTime(-cooldown) {}

    bool canCast(double elapsedTime) {
        // Check if the cooldown has passed
        const double epsilon = 1e-5;
        return std::abs(elapsedTime - lastCastTime) >= cooldownSeconds - epsilon;
    }

    void castAbility(double elapsedTime) {
        lastCastTime = elapsedTime;
    }
};

// Usage example:
DamageOverTimeAbility singleInstanceAbility(20, 5.0); // Deals 20 damage, cooldown of 5 seconds

// Inside the game loop:
if (singleInstanceAbility.canCast(elapsedTimeSeconds)) {
    singleInstanceAbility.castAbility(elapsedTimeSeconds);
    // Deal the single instance of damage
    dummy.takeDamage(singleInstanceAbility.damage);
    // Continue with other game logic
}