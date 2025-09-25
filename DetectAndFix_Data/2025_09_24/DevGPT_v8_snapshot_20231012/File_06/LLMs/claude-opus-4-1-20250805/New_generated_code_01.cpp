#include <iostream>
#include <limits>
#include <cmath>

int main() {
    int tt;
    
    // Validate initial input
    if (!(std::cin >> tt) || tt < 0 || tt > 1000000) {
        std::cerr << "Invalid test case count\n";
        return 1;
    }
    
    while (tt--) {
        int n, m, k, H;
        
        // Validate input with proper error handling
        if (!(std::cin >> n >> m >> k >> H)) {
            std::cerr << "Invalid input\n";
            return 1;
        }
        
        // Validate bounds and prevent divide by zero
        if (n < 0 || n > 1000000 || m < 1 || m > 1000000 || k <= 0 || k > 1000000) {
            std::cerr << "Input values out of valid range\n";
            return 1;
        }
        
        // Check for potential overflow before multiplication
        if (m > 1 && k > std::numeric_limits<int>::max() / (m - 1)) {
            std::cerr << "Potential integer overflow detected\n";
            return 1;
        }
        
        int ans = 0;
        long long max_diff = static_cast<long long>(k) * (m - 1);
        
        for (int i = 0; i < n; i++) {
            int h;
            if (!(std::cin >> h)) {
                std::cerr << "Invalid height input\n";
                return 1;
            }
            
            // Validate height is within reasonable bounds
            if (h < -1000000000 || h > 1000000000) {
                std::cerr << "Height value out of range\n";
                return 1;
            }
            
            // Safe modulo operation (k > 0 is guaranteed)
            // Using proper modulo for negative numbers
            auto safe_mod = [](int a, int b) -> int {
                int result = a % b;
                return (result < 0) ? result + b : result;
            };
            
            if (h != H && 
                safe_mod(h, k) == safe_mod(H, k) && 
                std::abs(static_cast<long long>(h) - H) <= max_diff) {
                ans++;
            }
        }
        
        std::cout << ans << '\n';
    }
    
    return 0;
}