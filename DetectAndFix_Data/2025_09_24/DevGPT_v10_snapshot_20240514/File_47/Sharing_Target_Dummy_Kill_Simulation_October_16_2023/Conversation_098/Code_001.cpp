class Ability {
public:
    double damage;
    double damageInterval;
    double durationSeconds;
    double cooldown;
    int priority;
    std::string name;
    double lastCastTime;
    double lastDamageTime;
    bool isActive;

    // Define a small epsilon value for comparisons
    static const double epsilon;

    Ability(double dmg, double interval, double duration, double cd, int pri, const std::string& n)
        : damage(dmg), damageInterval(interval), durationSeconds(duration), cooldown(cd), priority(pri), name(n),
          lastCastTime(-cd), lastDamageTime(-interval), isActive(false) {}

    bool canCast(double elapsedTime) {
        return !isActive && (elapsedTime - lastCastTime >= cooldown + epsilon);
    }

    bool canDealDamage(double elapsedTime) {
        return isActive && (elapsedTime - lastDamageTime >= damageInterval + epsilon);
    }

    void castAbility(double elapsedTime) {
        lastCastTime = elapsedTime;
        isActive = true;
    }

    void updateActivity(double elapsedTime) {
        if (elapsedTime - lastCastTime >= durationSeconds - epsilon) {
            isActive = false;
        }
    }
};

const double Ability::epsilon = 1e-5; // Adjust the epsilon value as needed