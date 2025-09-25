#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::vector<int> arr;
    constexpr std::size_t n = 10;

    // Replace predictable rand()/srand(time(0)) with <random>.
    // Note: std::mt19937 is not cryptographically secure. For security-critical randomness, use a CSPRNG.
    std::random_device rd;           // typically non-deterministic
    std::mt19937 gen(rd());          // seed PRNG from random_device
    std::uniform_int_distribution<int> dist(0, 99);

    arr.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        arr.push_back(dist(gen));
    }

    // Use std::sort (introsort) to avoid worst-case O(n^2) and deep recursion (fixes CWE-400/CWE-834/CWE-674).
    std::sort(arr.begin(), arr.end());

    for (std::size_t i = 0; i < n; ++i) {
        std::cout << arr[i] << (i + 1 == n ? '\n' : ' ');
    }

    return 0;
}