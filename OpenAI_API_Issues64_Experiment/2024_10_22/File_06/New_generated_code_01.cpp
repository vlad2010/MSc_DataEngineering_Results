#include <iostream>
#include <limits>

bool isValidInput(int value, int min, int max) {
    return value >= min && value <= max;
}

int main() {
    int tt;
    std::cin >> tt;

    // Validate the number of test cases
    if (!isValidInput(tt, 0, 1000)) { // Assume max 1000 test cases for this example
        std::cerr << "Invalid number of test cases.\n";
        return 1;
    }

    while (tt--) {
        int n, m, k, H;
        
        std::cin >> n >> m >> k >> H;
        if (std::cin.fail() || !isValidInput(n, 1, 100) || !isValidInput(m, 1, 1000) || k <= 0) {
            std::cerr << "Invalid input.\n";
            return 1;
        }

        int ans = 0;
        for (int i = 0; i < n; i++) {
            int h;
            std::cin >> h;

            if (std::cin.fail()) {
                std::cerr << "Input error.\n";
                return 1;
            }

            // Check for integer overflow and logic comparisons safely
            if (h != H && h % k == H % k) {
                long long diff = static_cast<long long>(h - H);
                long long maxDiff = static_cast<long long>(k) * (m - 1);
                if (std::abs(diff) <= maxDiff) {
                    ans += 1;
                }
            }
        }
        
        std::cout << ans << '\n';
    }

    return 0;
}