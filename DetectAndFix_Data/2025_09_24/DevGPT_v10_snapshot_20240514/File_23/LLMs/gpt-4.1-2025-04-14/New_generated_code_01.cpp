#include <iostream>
#include <limits>
#include <stdexcept>
#include <algorithm>

// Function to check if a, b, and c form a Pythagorean triplet
bool isPythagorean(int a, int b, int c) {
    // Pythagorean theorem: a^2 + b^2 = c^2
    return (a > 0 && b > 0 && c > 0 && a * a + b * b == c * c);
}

// Function to find the product of the Pythagorean triplet whose sum is given
long long findPythagoreanProduct(int sumOfSides) {
    if (sumOfSides < 12)
        return -1; // Not possible to form a Pythagorean triplet with sum < 12

    long long maxProduct = -1;

    // Using Euclid's formula to find Pythagorean triplets
    // Limit m to avoid integer overflow in m*m
    int max_m = static_cast<int>(std::sqrt(sumOfSides));
    for (int m = 2; m < max_m; ++m) {
        for (int n = 1; n < m; ++n) {
            // Check for potential overflow before calculation
            if (m > std::numeric_limits<int>::max() / m) continue;
            int m2 = m * m;
            int n2 = n * n;
            int a = m2 - n2;
            int b = 2 * m * n;
            int c = m2 + n2;

            // Check for overflow in a, b, c
            if (a <= 0 || b <= 0 || c <= 0) continue;

            // Check if the triplet is valid and its sum equals sumOfSides
            if (a + b + c == sumOfSides && isPythagorean(a, b, c)) {
                // Check for overflow in product
                long long product = static_cast<long long>(a) * b * c;
                maxProduct = std::max(maxProduct, product);
            }
        }
    }

    return maxProduct;
}

// Helper function to safely read an integer from std::cin
bool safeReadInt(const char* prompt, int& value, int minValue = 1, int maxValue = std::numeric_limits<int>::max()) {
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value < minValue || value > maxValue) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter an integer between " << minValue << " and " << maxValue << ".\n";
        } else {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return true;
        }
    }
}

int main() {
    int q;
    // Limit q to a reasonable number to prevent resource exhaustion
    const int MAX_Q = 1000;
    safeReadInt("How many times should this program be repeated: ", q, 1, MAX_Q);

    for (int i = 0; i < q; ++i) {
        int sumOfSides;
        // Limit sumOfSides to avoid overflow in calculations
        const int MAX_SUM = 1000000;
        safeReadInt("Enter the sum of sides: ", sumOfSides, 12, MAX_SUM);

        long long product = findPythagoreanProduct(sumOfSides);
        std::cout << "Product of Pythagorean triplet: " << product << std::endl;
    }

    return 0;
}