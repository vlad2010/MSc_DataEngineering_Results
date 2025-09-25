#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace std;

const long double PI = 3.14159265358979323846L;
const long double E = 2.71828182845904523536L;
const long long mod = 1000000007;

// Input validation helper
template<typename T>
bool safe_input(T& value, T min_val, T max_val) {
    if (!(cin >> value)) {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        return false;
    }
    if (value < min_val || value > max_val) {
        return false;
    }
    return true;
}

// Safe subtraction to prevent overflow
bool safe_subtract(int a, int b, int& result) {
    if (b > 0 && a < numeric_limits<int>::min() + b) {
        return false; // Would underflow
    }
    if (b < 0 && a > numeric_limits<int>::max() + b) {
        return false; // Would overflow
    }
    result = a - b;
    return true;
}

void solve() {
    const int MAX_N = 100000;  // Reasonable upper bound for array size
    const int MAX_VAL = 1000000000;  // Reasonable upper bound for values
    
    int n, m, k, h;
    
    // Input validation with bounds checking
    if (!safe_input(n, 1, MAX_N)) {
        cerr << "Invalid input for n" << endl;
        return;
    }
    
    if (!safe_input(m, 1, MAX_VAL)) {
        cerr << "Invalid input for m" << endl;
        return;
    }
    
    if (!safe_input(k, 1, MAX_VAL)) {  // k must be > 0 to avoid division by zero
        cerr << "Invalid input for k (must be positive)" << endl;
        return;
    }
    
    if (!safe_input(h, -MAX_VAL, MAX_VAL)) {
        cerr << "Invalid input for h" << endl;
        return;
    }
    
    vector<int> v;
    v.reserve(n);  // Pre-allocate for efficiency
    
    for (int i = 0; i < n; i++) {
        int val;
        if (!safe_input(val, -MAX_VAL, MAX_VAL)) {
            cerr << "Invalid input for array element " << i << endl;
            return;
        }
        
        // Safe subtraction and absolute value calculation
        int diff;
        if (!safe_subtract(val, h, diff)) {
            cerr << "Integer overflow detected in subtraction" << endl;
            return;
        }
        
        // Handle INT_MIN case for abs()
        if (diff == numeric_limits<int>::min()) {
            cerr << "Cannot take absolute value of INT_MIN" << endl;
            return;
        }
        
        v.push_back(abs(diff));
    }
    
    int ans = 0;
    
    for (int i = 0; i < n; i++) {
        if (v[i] == 0) {
            continue;
        }
        
        // Division by zero already prevented by input validation
        int remainder = v[i] % k;
        
        if (remainder == 0) {
            int quotient = v[i] / k;
            if (quotient < m) {
                // Check for overflow before incrementing
                if (ans == numeric_limits<int>::max()) {
                    cerr << "Answer overflow" << endl;
                    return;
                }
                ans++;
            }
        }
    }
    
    cout << ans << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    try {
        solve();
    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}