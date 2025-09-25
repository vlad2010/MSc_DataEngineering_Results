#include <iostream>
#include <limits>
#include <cstdlib> // for std::abs

using namespace std;

// Helper function to safely multiply two integers and check for overflow
bool safe_multiply(int a, int b, int& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    if (a > 0) {
        if (b > 0) {
            if (a > (std::numeric_limits<int>::max() / b)) return false;
        } else {
            if (b < (std::numeric_limits<int>::min() / a)) return false;
        }
    } else {
        if (b > 0) {
            if (a < (std::numeric_limits<int>::min() / b)) return false;
        } else {
            if (a != 0 && b < (std::numeric_limits<int>::max() / a)) return false;
        }
    }
    result = a * b;
    return true;
}

int main() {
    int tt;
    if (!(cin >> tt) || tt <= 0 || tt > 10000) { // Arbitrary upper bound for safety
        cerr << "Invalid number of test cases.\n";
        return 1;
    }
    while (tt--) {
        int n, m, k, H;
        if (!(cin >> n >> m >> k >> H)) {
            cerr << "Invalid input for n, m, k, H.\n";
            return 1;
        }
        if (n <= 0 || n > 100000) { // Arbitrary upper bound for safety
            cerr << "Invalid value for n.\n";
            return 1;
        }
        if (m <= 0 || k <= 0) {
            cerr << "Invalid value for m or k.\n";
            return 1;
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int h;
            if (!(cin >> h)) {
                cerr << "Invalid input for h.\n";
                return 1;
            }
            // Check for overflow in k * (m - 1)
            int km1;
            if (!safe_multiply(k, m - 1, km1)) {
                cerr << "Overflow in k * (m - 1).\n";
                return 1;
            }
            // Check for overflow in abs(h - H)
            int diff;
            if (__builtin_sub_overflow(h, H, &diff)) {
                cerr << "Overflow in h - H.\n";
                return 1;
            }
            int abs_diff = std::abs(diff);
            if (h != H && h % k == H % k && abs_diff <= km1) {
                ans += 1;
            }
        }
        cout << ans << '\n';
    }
    return 0;
}