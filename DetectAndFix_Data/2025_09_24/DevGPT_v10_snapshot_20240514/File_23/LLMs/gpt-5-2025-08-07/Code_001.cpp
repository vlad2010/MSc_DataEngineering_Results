#include <iostream>

// Function to check if a, b, and c form a Pythagorean triplet
bool isPythagorean(int a, int b, int c) {
    // Pythagorean theorem: a^2 + b^2 = c^2
    return (a * a + b * b == c * c);
}

// Function to find the product of the Pythagorean triplet whose sum is given
long long findPythagoreanProduct(int sumOfSides) {
    if (sumOfSides < 12)
        return -1; // Not possible to form a Pythagorean triplet with sum < 12
    
    long long maxProduct = -1;
    
    // Using Euclid's formula to find Pythagorean triplets
    for (int m = 2; m * m < sumOfSides; ++m) {
        for (int n = 1; n < m; ++n) {
            // Euclid's formula to generate Pythagorean triplets:
            // a = m^2 - n^2, b = 2mn, c = m^2 + n^2
            int a = m * m - n * n;
            int b = 2 * m * n;
            int c = m * m + n * n;
            
            // Check if the triplet is valid and its sum equals sumOfSides
            if (a + b + c == sumOfSides && isPythagorean(a, b, c)) {
                long long product = a * b * c;
                maxProduct = std::max(maxProduct, product);
            }
        }
    }
    
    return maxProduct;
}

int main() {
    int q;
    std::cout << "How many times should this program be repeated: ";
    std::cin >> q;

    for (int i = 0; i < q; ++i) {
        int sumOfSides;
        std::cout << "Enter the sum of sides: ";
        std::cin >> sumOfSides;

        long long product = findPythagoreanProduct(sumOfSides);
        std::cout << "Product of Pythagorean triplet: " << product << std::endl;
    }

    return 0;
}