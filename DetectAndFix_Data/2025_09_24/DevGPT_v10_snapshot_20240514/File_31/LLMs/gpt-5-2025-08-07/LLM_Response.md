Analysis (security-focused)

- Algorithmic complexity and recursion depth risks (CWE-400: Uncontrolled Resource Consumption; CWE-834: Excessive Iteration; CWE-674: Uncontrolled Recursion)
  - Picking the last element as pivot with Lomuto partition makes quicksort degrade to O(n^2) time and O(n) recursion depth on already-sorted or adversarial input. This can be exploited to cause CPU and stack exhaustion (DoS), especially if the array size or content is attacker-controlled.
  - Fixes:
    - Prefer std::sort, which uses introsort (quicksort + heapsort fallback), guaranteeing O(n log n) worst-case and bounded stack depth.
    - If you must keep custom quicksort: use randomized or median-of-three pivot selection and perform tail recursion elimination or an iterative approach with an explicit stack; optionally add an introspective depth limit and fall back to heapsort.

- Potential improper index handling if reused as a library function (CWE-129: Improper Validation of Array Index; CWE-681: Incorrect Conversion Between Numeric Types; CWE-787/125: Out-of-bounds Write/Read)
  - The quicksort/partition API takes int indices. In general use, signed/unsigned conversions to size_t on vector::operator[] can underflow/overflow if callers pass invalid values, leading to OOB access. Current main is safe but the functions are not defensive.
  - Fixes:
    - Use iterators or size_t and validate bounds before indexing, or avoid custom indexing altogether by using std::sort which operates on iterators.

- Predictable pseudo-random numbers in srand/rand (CWE-338: Use of Cryptographically Weak PRNG)
  - srand(time(0)) + rand() is predictable and not suitable for any security-relevant randomness. Even for non-security purposes it’s low quality and has modulo bias with rand() % 100.
  - Fix:
    - Use <random> with std::random_device to seed a generator and std::uniform_int_distribution. If randomness is security-critical, use a CSPRNG (e.g., std::random_device directly per draw, or platform CSPRNG APIs).

- Minor hardening
  - Avoid using namespace std; to reduce namespace pollution (not a CWE but good practice).

Fixed code (uses std::sort to eliminate the algorithmic DoS risk and replaces rand/srand)

```cpp
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
```

Notes if you must keep a custom quicksort:
- Use iterators instead of int indices to prevent signed/unsigned mistakes (mitigates CWE-129/681).
- Randomize or use median-of-three pivot selection.
- Apply tail recursion elimination or an iterative approach to bound stack depth.
- Consider an introspective quicksort that falls back to heapsort when recursion depth exceeds 2*log2(n).