#include <bits/stdc++.h>
using namespace std;

// Safe multiply: computes a*b into out; returns true if overflow occurred.
static bool mul_overflow_ll(long long a, long long b, long long& out) {
  __int128 tmp = static_cast<__int128>(a) * static_cast<__int128>(b);
  if (tmp > std::numeric_limits<long long>::max() || tmp < std::numeric_limits<long long>::min()) {
    return true;
  }
  out = static_cast<long long>(tmp);
  return false;
}

// Compute |a - b| safely using 128-bit intermediate, clamp to LLONG_MAX if needed.
static long long safe_abs_diff_ll(long long a, long long b) {
  __int128 d = static_cast<__int128>(a) - static_cast<__int128>(b);
  if (d < 0) d = -d;
  if (d > std::numeric_limits<long long>::max()) {
    return std::numeric_limits<long long>::max();
  }
  return static_cast<long long>(d);
}

// Normalize modulo to non-negative result for positive modulus k (>0).
static long long mod_norm(long long x, long long k) {
  long long r = x % k;
  if (r < 0) r += k;
  return r;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  long long tt;
  if (!(cin >> tt) || tt < 0) {
    cerr << "Invalid number of test cases\n";
    return 1;
  }

  while (tt--) {
    long long n, m, k, H;
    if (!(cin >> n >> m >> k >> H)) {
      cerr << "Failed to read n, m, k, H\n";
      return 1;
    }

    // Input validation to prevent divide-by-zero, negative lengths, etc.
    if (n < 0) {
      cerr << "n must be non-negative\n";
      return 1;
    }
    if (m <= 0) {
      cerr << "m must be positive\n";
      return 1;
    }
    if (k <= 0) {
      cerr << "k must be positive (prevents modulo/divide by zero)\n";
      return 1;
    }

    // Precompute safe bound = k * (m - 1), handling potential overflow.
    long long bound = 0;
    bool ovf = mul_overflow_ll(k, m - 1, bound);
    if (ovf) {
      // If it overflows, it's effectively larger than any practical |h - H|.
      bound = std::numeric_limits<long long>::max();
    }

    long long ans = 0;
    for (long long i = 0; i < n; ++i) {
      long long h;
      if (!(cin >> h)) {
        cerr << "Failed to read h[" << i << "]\n";
        return 1;
      }

      // Normalize remainders to avoid negative modulo surprises.
      long long rH = mod_norm(H, k);
      long long rh = mod_norm(h, k);

      long long diff = safe_abs_diff_ll(h, H);

      if (h != H && rh == rH && diff <= bound) {
        // Cap ans against overflow just in case (n could be very large).
        if (ans < std::numeric_limits<long long>::max()) {
          ++ans;
        }
      }
    }

    cout << ans << '\n';
  }

  return 0;
}