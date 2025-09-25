#include <iostream>
#include <ctime>
#include <cstdlib>

// Define a target dummy with health
class TargetDummy {
public:
    int health;

    TargetDummy(int initialHealth) : health(initialHealth) {}
    bool isAlive() { return health > 0; }
    void takeDamage(int damage) { health -= damage; }
};

int main() {
    // Initialize random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    int damagePerHit = 20; // Damage per hit
    double tickDuration = 0.1; // 1 tick = 0.1 seconds

    int ticks = 0;

    while (dummy.isAlive()) {
        // Simulate hitting the dummy
        dummy.takeDamage(damagePerHit);
        ticks++;

        // You can insert your game logic here

        // Display the dummy's current health
        std::cout << "Tick " << ticks << ": Dummy Health: " << dummy.health << std::endl;
    }

    // Calculate and display the time taken to kill the dummy in seconds
    double elapsedTime = ticks * tickDuration;
    std::cout << "Target dummy killed in " << elapsedTime << " seconds." << std::endl;

    return 0;
}