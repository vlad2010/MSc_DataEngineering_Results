Ability(double dmg, double interval, double duration, double cd, int pri, const std::string& n)
    : damage(dmg), damageInterval(interval), durationSeconds(duration), cooldown(cd), priority(pri), name(n),
      lastCastTime(-duration), lastDamageTime(-interval), isActive(false) {}