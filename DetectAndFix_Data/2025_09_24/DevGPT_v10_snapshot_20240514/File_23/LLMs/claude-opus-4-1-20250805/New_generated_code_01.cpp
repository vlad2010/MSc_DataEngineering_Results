#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

// Safe multiplication with overflow check
bool safe_multiply(long long a, long long b, long long& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    
    // Check for overflow before multiplication
    if (a > 0 && b > 0 && a > std::numeric_limits<long long>::max() / b) {
        return false;
    }
    if (a < 0 && b < 0 && a < std::numeric_limits<long long>::max() / b) {
        return false;
    }
    if (a > 0 && b < 0 && b < std::numeric_limits<long long>::min() / a) {
        return false;
    }
    if (a < 0 && b > 0 && a < std::numeric_limits<long long>::min() / b) {
        return false;
    }
    
    result = a * b;
    return true;
}

// Safe addition with overflow check
bool safe_add(long long a, long long b, long long& result) {
    if (b > 0 && a > std::numeric_limits<long long>::max() - b) {
        return false;
    }
    if (b < 0 && a < std::numeric_limits<long long>::min() - b) {
        return false;
    }
    result = a + b;
    return true;
}

// Function to check if a, b, and c form a Pythagorean triplet with overflow protection
bool isPythagorean(long long a, long long b, long long c) {
    long long a_squared, b_squared, c_squared, sum_ab;
    
    if (!safe_multiply(a, a, a_squared) || 
        !safe_multiply(b, b, b_squared) || 
        !safe_multiply(c, c, c_squared)) {
        return false; // Overflow occurred
    }
    
    if (!safe_add(a_squared, b_squared, sum_ab)) {
        return false; // Overflow occurred
    }
    
    return (sum_ab == c_squared);
}

// Function to find the product of the Pythagorean triplet whose sum is given
long long findPythagoreanProduct(int sumOfSides) {
    // Input validation
    if (sumOfSides < 12 || sumOfSides > 10000) {
        return -1; // Reasonable bounds for the problem
    }
    
    long long maxProduct = -1;
    
    // Limit the search space to prevent DoS
    int maxM = std::min(static_cast<int>(std::sqrt(sumOfSides)), 1000);
    
    // Using Euclid's formula to find Pythagorean triplets
    for (int m = 2; m < maxM && m * m < sumOfSides; ++m) {
        for (int n = 1; n < m && n < 1000; ++n) {
            // Use long long to prevent overflow in intermediate calculations
            long long m_ll = m;
            long long n_ll = n;
            
            long long m_squared, n_squared;
            if (!safe_multiply(m_ll, m_ll, m_squared) || 
                !safe_multiply(n_ll, n_ll, n_squared)) {
                continue; // Skip if overflow
            }
            
            long long a, b, c;
            
            // a = m^2 - n^2
            if (m_squared < n_squared) {
                continue; // Invalid triplet
            }
            a = m_squared - n_squared;
            
            // b = 2mn
            long long two_mn;
            if (!safe_multiply(2LL, m_ll, two_mn)) {
                continue;
            }
            if (!safe_multiply(two_mn / 2, n_ll, b)) {
                continue;
            }
            
            // c = m^2 + n^2
            if (!safe_add(m_squared, n_squared, c)) {
                continue;
            }
            
            // Check sum
            long long sum_ab, sum_abc;
            if (!safe_add(a, b, sum_ab) || !safe_add(sum_ab, c, sum_abc)) {
                continue;
            }
            
            // Check if the triplet is valid and its sum equals sumOfSides
            if (sum_abc == sumOfSides && isPythagorean(a, b, c)) {
                long long product_ab, product_abc;
                if (safe_multiply(a, b, product_ab) && 
                    safe_multiply(product_ab, c, product_abc)) {
                    maxProduct = std::max(maxProduct, product_abc);
                }
            }
        }
    }
    
    return maxProduct;
}

// Safe input reading function
bool readInteger(int& value, const std::string& prompt, int minVal, int maxVal) {
    std::cout << prompt;
    
    if (!(std::cin >> value)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cerr << "Invalid input. Please enter a valid integer.\n";
        return false;
    }
    
    if (value < minVal || value > maxVal) {
        std::cerr << "Value must be between " << minVal << " and " << maxVal << ".\n";
        return false;
    }
    
    return true;
}

int main() {
    int q;
    
    // Input validation for number of iterations
    if (!readInteger(q, "How many times should this program be repeated (1-100): ", 1, 100)) {
        return 1;
    }

    for (int i = 0; i < q; ++i) {
        int sumOfSides;
        
        // Input validation for sum of sides
        if (!readInteger(sumOfSides, "Enter the sum of sides (12-10000): ", 12, 10000)) {
            std::cerr << "Skipping this iteration due to invalid input.\n";
            continue;
        }

        long long product = findPythagoreanProduct(sumOfSides);
        
        if (product == -1) {
            std::cout << "No Pythagorean triplet found with sum " << sumOfSides << std::endl;
        } else {
            std::cout << "Product of Pythagorean triplet: " << product << std::endl;
        }
    }

    return 0;
}