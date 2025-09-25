#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib> // for std::abs

int safe_read_int(const char* prompt = nullptr) {
    int x;
    while (true) {
        if (prompt) std::cout << prompt;
        std::cin >> x;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cerr << "Invalid input. Please enter an integer." << std::endl;
        } else {
            break;
        }
    }
    return x;
}

void solve()
{
    int n = safe_read_int("Enter n (number of elements, positive integer): ");
    if (n <= 0 || n > 1000000) { // Arbitrary upper bound for safety
        std::cerr << "Invalid value for n." << std::endl;
        return;
    }
    int m = safe_read_int("Enter m (positive integer): ");
    if (m <= 0) {
        std::cerr << "Invalid value for m." << std::endl;
        return;
    }
    int k = safe_read_int("Enter k (non-zero integer): ");
    if (k == 0) {
        std::cerr << "k cannot be zero." << std::endl;
        return;
    }
    int h = safe_read_int("Enter h (integer): ");

    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = safe_read_int("Enter element: ");
        // Check for overflow in subtraction
        long long diff = static_cast<long long>(v[i]) - static_cast<long long>(h);
        if (diff < std::numeric_limits<int>::min() || diff > std::numeric_limits<int>::max()) {
            std::cerr << "Overflow detected in abs(v[i] - h)." << std::endl;
            return;
        }
        v[i] = std::abs(static_cast<int>(diff));
    }

    int ans = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i] == 0)
            continue;
        int x = v[i] % k;
        if (x == 0) {
            int div = v[i] / k;
            if (div < m) {
                ans++;
            }
        }
    }
    std::cout << ans << std::endl;
}

int main() {
    solve();
    return 0;
}