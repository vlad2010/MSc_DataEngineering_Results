#include <iostream>
#include <random>

float getRandomFloat(float min, float max) {
    std::random_device rd;                          // Obtain a random seed from the hardware
    std::mt19937 eng(rd());                         // Seed the generator
    std::uniform_real_distribution<float> distr(min, max); // Define the range

    return distr(eng);                              // Generate and return a random float
}

int main() {
    // Example usage
    float minValue = 1.0;
    float maxValue = 10.0;
    float randomValue = getRandomFloat(minValue, maxValue);
    
    std::cout << "Random float value between " << minValue << " and " << maxValue << ": " << randomValue << std::endl;

    return 0;
}
