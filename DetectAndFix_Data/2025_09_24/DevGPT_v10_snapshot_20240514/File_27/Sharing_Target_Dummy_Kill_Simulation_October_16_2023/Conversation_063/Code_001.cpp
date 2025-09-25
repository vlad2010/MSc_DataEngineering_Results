if (instantDamageAbility.canDealDamage(elapsedTimeSeconds)) {
    // Deal the one-time damage
    dummy.takeDamage(static_cast<int>(instantDamageAbility.damagePerTick));
    instantDamageAbility.lastDamageTime = elapsedTimeSeconds;
    // Continue with other game logic
}