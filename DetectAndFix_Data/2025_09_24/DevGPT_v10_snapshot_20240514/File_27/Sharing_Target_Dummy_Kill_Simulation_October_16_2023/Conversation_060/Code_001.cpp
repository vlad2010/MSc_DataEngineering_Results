DamageOverTimeAbility(double damage, double cooldown, double duration, int pri)
    : damagePerTick(damage), cooldownSeconds(cooldown), durationSeconds(duration),
      lastCastTime(-duration), lastDamageTime(-cooldown), isActive(false), priority(pri) {}