Ability(double damage, double interval, double duration, double cooldown, int pri)
    : damagePerTick(damage), damageInterval(interval), durationSeconds(duration),
      lastCastTime(-cooldown), lastDamageTime(-interval), individualCooldown(cooldown), isActive(false), priority(pri) {}