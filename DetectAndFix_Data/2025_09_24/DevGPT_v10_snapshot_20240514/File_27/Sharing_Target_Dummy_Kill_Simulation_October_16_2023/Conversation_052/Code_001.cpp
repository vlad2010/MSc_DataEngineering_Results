void updateActivity(double elapsedTime) {
    // Use an epsilon (tolerance) to check for almost equality
    const double epsilon = 1e-5; // Adjust the value as needed
    if (std::abs(elapsedTime - lastCastTime) >= durationSeconds - epsilon) {
        isActive = false;
    }
}