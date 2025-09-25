bool canCast(double elapsedTime, double sharedCooldown) {
    const double epsilon = 1e-5;
    return !isActive && (elapsedTime - lastCastTime >= cooldown - epsilon) && (sharedCooldown <= epsilon);
}