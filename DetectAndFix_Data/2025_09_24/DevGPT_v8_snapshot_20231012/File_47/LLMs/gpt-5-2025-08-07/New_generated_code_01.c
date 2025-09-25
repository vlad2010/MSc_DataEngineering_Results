#include <stdio.h>
#include <stddef.h>

/*
 * Robust ARRAY_LEN macro:
 * - Computes the number of elements in a true array.
 * - Triggers a compile-time error if passed a pointer (guards against CWE-467/CWE-131).
 * Works on GCC/Clang; falls back to a plain sizeof-based version elsewhere.
 */
#if defined(__GNUC__) || defined(__clang__)
#define ARRAY_LEN(a) \
    ( (void)sizeof(char[1 - 2*!!__builtin_types_compatible_p(__typeof__(a), __typeof__(&(a)[0]))]), \
      sizeof(a) / sizeof((a)[0]) )
#else
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#endif

int main(void) {
    int numbers[] = {1, 2, 3, 4, 5};

    size_t n = ARRAY_LEN(numbers);  // Compile-time safe length
    for (size_t i = 0; i < n; ++i) {
        // Constant format string (safe with printf; avoids CWE-134)
        printf("%d%s", numbers[i], (i + 1 == n) ? "\n" : " ");
    }

    return 0;
}