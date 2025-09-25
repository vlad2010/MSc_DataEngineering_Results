class Ability {
public:
    double damage;         // Damage per cast
    double damageInterval; // Interval between damage ticks
    double durationSeconds; // Duration of the ability
    double cooldown;       // Cooldown before the ability can be cast again
    int priority;          // Priority of the ability
    std::string name;      // Name of the ability
    double lastCastTime;   // Track when the ability was last cast
    double lastDamageTime; // Track when the ability last dealt damage
    bool isActive;         // Indicate if the ability is active

    Ability(double dmg, double interval, double duration, double cd, int pri, const std::string& n)
        : damage(dmg), damageInterval(interval), durationSeconds(duration), cooldown(cd), priority(pri), name(n),
          lastCastTime(-cd), lastDamageTime(-interval), isActive(false) {}
};