#include <iostream>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <climits>

// Safe multiplication (int64) using 128-bit intermediate to detect overflow.
static inline bool mulOverflowLL(int64_t a, int64_t b, int64_t& out) {
#if defined(__SIZEOF_INT128__)
    __int128 r = static_cast<__int128>(a) * static_cast<__int128>(b);
    if (r < LLONG_MIN || r > LLONG_MAX) return true;
    out = static_cast<int64_t>(r);
    return false;
#else
    // Portable fallback (conservative); avoids UB by checking bounds before multiply.
    if (a == 0 || b == 0) { out = 0; return false; }
    if (a > 0) {
        if (b > 0) {
            if (a > LLONG_MAX / b) return true;
        } else { // b < 0
            if (b < LLONG_MIN / a) return true;
        }
    } else { // a < 0
        if (b > 0) {
            if (a < LLONG_MIN / b) return true;
        } else { // b < 0
            if (a != 0 && -a > LLONG_MAX / -b) return true;
        }
    }
    out = a * b;
    return false;
#endif
}

static inline bool mul3OverflowLL(int64_t a, int64_t b, int64_t c, int64_t& out) {
    int64_t tmp;
    if (mulOverflowLL(a, b, tmp)) return true;
    if (mulOverflowLL(tmp, c, out)) return true;
    return false;
}

// Safe Pythagorean check with 128-bit intermediates to avoid overflow.
static inline bool isPythagoreanLL(int64_t a, int64_t b, int64_t c) {
    if (a <= 0 || b <= 0 || c <= 0) return false;
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
#if defined(__SIZEOF_INT128__)
    __int128 aa = static_cast<__int128>(a) * a;
    __int128 bb = static_cast<__int128>(b) * b;
    __int128 cc = static_cast<__int128>(c) * c;
    return (aa + bb) == cc;
#else
    // Fallback using long double (less strict, but acceptable under bounds we enforce).
    long double aa = static_cast<long double>(a) * a;
    long double bb = static_cast<long double>(b) * b;
    long double cc = static_cast<long double>(c) * c;
    return fabsl((aa + bb) - cc) < 0.5L; // tolerance for rounding
#endif
}

// Find the product of a Pythagorean triplet (a, b, c) with a + b + c = sumOfSides.
// Uses Euclid's formula with scaling factor k and overflow checks.
// Returns true on success and sets outProduct. Returns false if not found or product would overflow int64.
bool findPythagoreanProduct(int64_t sumOfSides, int64_t& outProduct) {
    if (sumOfSides < 12) return false;

    // m limit derived from sum = 2*k*m*(m+n) >= 2*m*(m+1), so m <= sqrt(sum/2).
    long double s2 = static_cast<long double>(sumOfSides) / 2.0L;
    if (s2 <= 0.0L) return false;
    int64_t mLimit = static_cast<int64_t>(std::floor(std::sqrt(s2)));
    if (mLimit < 2) return false;

    int64_t maxProduct = LLONG_MIN;
    bool found = false;

    for (int64_t m = 2; m <= mLimit; ++m) {
        for (int64_t n = 1; n < m; ++n) {
            // Generate only primitive triplets: coprime and opposite parity.
            if (((m - n) & 1) == 0) continue;          // same parity -> skip
            if (std::gcd(m, n) != 1) continue;         // not coprime -> skip

            // Base primitive triplet
            int64_t mm = m * m;                         // safe for our bounded m
            int64_t nn = n * n;
            int64_t a = mm - nn;
            int64_t b = 2 * m * n;
            int64_t c = mm + nn;

            // Base sum is 2*m*(m+n) but a+b+c is fine and small under our bounds.
            int64_t baseSum = a + b + c;
            if (baseSum <= 0) continue; // defensive

            if (sumOfSides % baseSum != 0) continue;
            int64_t k = sumOfSides / baseSum;

            // Scale (a, b, c) by k with overflow checks.
            int64_t A, B, C;
            if (mulOverflowLL(a, k, A)) continue;
            if (mulOverflowLL(b, k, B)) continue;
            if (mulOverflowLL(c, k, C)) continue;

            if (!isPythagoreanLL(A, B, C)) continue; // defensive check

            int64_t product;
            if (mul3OverflowLL(A, B, C, product)) {
                // Product would overflow 64-bit; skip this candidate
                continue;
            }

            if (!found || product > maxProduct) {
                maxProduct = product;
                found = true;
            }
        }
    }

    if (!found) return false;
    outProduct = maxProduct;
    return true;
}

// Robust input reader with bounds.
bool readBoundedInt64(const char* prompt, int64_t minVal, int64_t maxVal, int64_t& out) {
    for (;;) {
        std::cout << prompt;
        if (!(std::cin >> out)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cerr << "Invalid number. Please try again.\n";
            continue;
        }
        if (out < minVal || out > maxVal) {
            std::cerr << "Value must be between " << minVal << " and " << maxVal << ". Please try again.\n";
            continue;
        }
        return true;
    }
}

int main() {
    // Reasonable safeguards to mitigate DoS via excessive iteration.
    // Adjust these as appropriate for your environment.
    const int64_t MAX_Q   = 1000;        // Max number of queries
    const int64_t MAX_SUM = 10000000;    // Max sum to keep runtime and values safe

    int64_t q = 0;
    if (!readBoundedInt64("How many times should this program be repeated: ", 1, MAX_Q, q)) {
        return 1;
    }

    for (int64_t i = 0; i < q; ++i) {
        int64_t sumOfSides = 0;
        if (!readBoundedInt64("Enter the sum of sides: ", 12, MAX_SUM, sumOfSides)) {
            return 1;
        }

        int64_t product = 0;
        if (findPythagoreanProduct(sumOfSides, product)) {
            std::cout << "Product of Pythagorean triplet: " << product << std::endl;
        } else {
            std::cout << "No valid Pythagorean triplet found within 64-bit range for that sum." << std::endl;
        }
    }

    return 0;
}