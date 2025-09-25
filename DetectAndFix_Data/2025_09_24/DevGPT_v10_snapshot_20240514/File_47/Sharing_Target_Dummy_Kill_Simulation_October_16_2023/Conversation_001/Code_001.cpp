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
    srand(static_cast<unsigned int>(time(nullptr));

    // Create a target dummy with an initial health value
    TargetDummy dummy(100); // You can set the initial health as per your requirements

    int damagePerHit = 20; // Damage per hit

    int hits = 0;
    std::clock_t start = std::clock();

    while (dummy.isAlive()) {
        // Simulate hitting the dummy
        dummy.takeDamage(damagePerHit);
        hits++;

        // You can introduce some delay here to simulate a game tick or frame
        // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Add #include <thread> and <chrono> for this

        // Display the dummy's current health
        std::cout << "Dummy Health: " << dummy.health << std::endl;
    }

    std::clock_t end = std::clock();
    double elapsedSeconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Calculate and display the time taken to kill the dummy
    std::cout << "Target dummy killed in " << elapsedSeconds << " seconds." << std::endl;
    std::cout << "Number of hits: " << hits << std::endl;

    return 0;
}