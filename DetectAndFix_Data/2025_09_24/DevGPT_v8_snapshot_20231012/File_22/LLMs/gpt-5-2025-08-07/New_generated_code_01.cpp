#include <iostream>
#include <vector>
#include <limits>
#include <cstdint>

static inline bool safeReadLL(long long &x) {
    if (!(std::cin >> x)) return false;
    return true;
}

// Safely compute |a - b| without signed overflow.
// Uses 128-bit intermediate when available; otherwise falls back with a conservative clamp.
static inline unsigned long long abs_diff_no_overflow(long long a, long long b) {
#if defined(__SIZEOF_INT128__)
    __int128 d = static_cast<__int128>(a) - static_cast<__int128>(b);
    if (d < 0) d = -d;
    return static_cast<unsigned long long>(d);
#else
    // Fallback: compute in 64-bit; guard the LLONG_MIN corner case.
    long long d = a - b;
    if (d == std::numeric_limits<long long>::min()) {
        // Clamp to LLONG_MAX to avoid UB in later abs; this preserves non-negativity and avoids crash.
        return static_cast<unsigned long long>(std::numeric_limits<long long>::max());
    }
    return static_cast<unsigned long long>(d >= 0 ? d : -d);
#endif
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    long long n_ll, m_ll, k_ll, h_ll;
    if (!safeReadLL(n_ll) || !safeReadLL(m_ll) || !safeReadLL(k_ll) || !safeReadLL(h_ll)) {
        std::cerr << "Input error\n";
        return 1;
    }

    // Validate untrusted inputs
    const long long MAX_N = 1'000'000; // Mitigate DoS via huge allocation. Tune as needed.
    if (n_ll < 0 || n_ll > MAX_N) {
        std::cerr << "Invalid n\n";
        return 1;
    }
    if (m_ll < 0) {
        std::cerr << "Invalid m\n";
        return 1;
    }
    if (k_ll <= 0) {
        // Original code cannot handle k == 0 (mod/div by zero). For k <= 0, define safe behavior.
        // Here we choose to return 0 as no valid steps can be computed safely.
        std::cout << 0 << '\n';
        return 0;
    }

    const size_t n = static_cast<size_t>(n_ll);
    const unsigned long long k = static_cast<unsigned long long>(k_ll);
    const long long h = h_ll;
    const unsigned long long m = static_cast<unsigned long long>(m_ll);

    std::vector<unsigned long long> v(n);

    for (size_t i = 0; i < n; ++i) {
        long long t;
        if (!safeReadLL(t)) {
            std::cerr << "Input error\n";
            return 1;
        }
        // Compute absolute distance safely
        v[i] = abs_diff_no_overflow(t, h);
    }

    unsigned long long ans = 0;
    for (size_t i = 0; i < n; ++i) {
        if (v[i] == 0ULL) continue;
        if (v[i] % k == 0ULL) {
            unsigned long long steps = v[i] / k;
            if (steps < m) {
                ++ans;
            }
        }
    }

    std::cout << ans << '\n';
    return 0;
}