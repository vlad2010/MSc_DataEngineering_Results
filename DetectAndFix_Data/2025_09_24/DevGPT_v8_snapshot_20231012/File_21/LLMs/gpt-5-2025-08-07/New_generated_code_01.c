#include <stdio.h>
#include <string.h>
#include <stddef.h>

/*
 * Constant-time string equality for secrets.
 * - Compares up to the max length, padding the shorter with 0.
 * - Accumulates differences without early exit.
 * - Returns 1 if equal, 0 otherwise.
 * Note: For true binary secrets (may contain NUL), pass explicit lengths.
 */
static int constant_time_str_equal(const char *a, size_t a_len,
                                   const char *b, size_t b_len) {
    size_t max = (a_len > b_len) ? a_len : b_len;
    unsigned char diff = 0;

    for (size_t i = 0; i < max; ++i) {
        unsigned char ca = (i < a_len) ? (unsigned char)a[i] : 0u;
        unsigned char cb = (i < b_len) ? (unsigned char)b[i] : 0u;
        diff |= (unsigned char)(ca ^ cb);
    }

    /* Also ensure lengths are equal without branching */
    diff |= (unsigned char)((a_len ^ b_len) != 0);

    return diff == 0;
}

/*
 * Safe bounded strcmp-like behavior for non-secret strings that may come
 * from fixed-size buffers. Ensures we never read past the buffers.
 * Returns negative/zero/positive similar to strcmp.
 */
static int safe_strcmp_bounded(const char *a, size_t a_cap,
                               const char *b, size_t b_cap) {
    size_t a_len = strnlen(a, a_cap);
    size_t b_len = strnlen(b, b_cap);

    /* Compare up to min(len)+1 so length differences are detected */
    size_t n = (a_len < b_len ? a_len : b_len) + 1;
    return strncmp(a, b, n);
}

int main(void) {
    /* Example data (as in your snippet) */
    char str1[] = "Hello";
    char str2[] = "World";

    /* If comparing non-secret strings and you want strcmp semantics,
       use a bounded variant to avoid potential OOB if inputs are from fixed buffers. */
    {
        int result = safe_strcmp_bounded(str1, sizeof(str1), str2, sizeof(str2));
        printf("Non-secret bounded compare (strcmp-like) result: %d\n", result);
    }

    /* If these were secrets (passwords, tokens, HMACs encoded as strings), avoid strcmp. */
    {
        int equal = constant_time_str_equal(str1, strlen(str1), str2, strlen(str2));
        /* Do not expose detailed comparison results in sensitive paths; just equal/not equal */
        printf("Secret-safe equal? %s\n", equal ? "yes" : "no");
    }

    return 0;
}