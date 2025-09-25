Below is a security review of your Base64 decoder and a corrected, hardened implementation.

Key issues and CWE classifications
- Out-of-bounds read (CWE-125) and inconsistent API use:
  - The function accepts src_len but ignores it and uses while (*src) and base64dec_getc, which read until NUL. If src is not NUL-terminated, this can read past the provided buffer.
- Heap-based buffer overflow (CWE-122) and incorrect buffer size calculation (CWE-131) driven by integer overflow (CWE-190):
  - Output size is computed from src_len, but decoding is driven by actual NUL-terminated content, leading to under-allocation if the real input is longer than src_len. Also, src_len/4*3 + 1 can overflow for large inputs.
- Improper input validation (CWE-20) and improper initialization (CWE-665):
  - base64_digits is a static char array default-initialized to 0 for all unspecified indexes. The code checks for -1 to detect invalid symbols, but no entry is ever set to -1, so invalid characters pass as 0 (“A”).
  - Only a and b are checked; c and d (and padding rules) are not robustly validated.
  - base64dec_getc returns '=' at EOF, fabricating padding, which breaks validation.
- Incorrect handling of whitespace and padding:
  - isprint-based skipping is wrong. Base64 allows ignoring whitespace (isspace), not all non-printables, and certainly should not auto-insert '='.

How the fix addresses this
- Reads are strictly bounded by src_len; no dependency on a terminating NUL.
- Robust, overflow-checked allocation using a safe worst-case bound: out_cap = 3 * ((src_len + 3) / 4) + 1, with guard checks to avoid size_t overflow.
- Proper Base64 value mapping and validation:
  - No reliance on a wrongly initialized lookup table; instead, a small function computes the Base64 value and returns -1 for invalid characters.
  - All four characters in each quartet are validated, including strict padding rules. After '=' padding, only whitespace is permitted.
- Whitespace handling uses isspace only.
- Writes are bounds-checked against the allocated buffer.
- Still returns a NUL-terminated buffer for compatibility, but note that Base64 may decode to binary with embedded NULs; returning the decoded length would be more robust (design note).

Fixed code (single fragment)
```c
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>  /* for SIZE_MAX */
#include <string.h>

/* Map a Base64 character to its 6-bit value; return -1 if invalid.
   Supports standard Base64 (+, /). If you want URL-safe, add '-'->62 and '_'->63. */
static inline int b64_val(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return (int)(ch - 'A');              /* 0..25 */
    if (ch >= 'a' && ch <= 'z') return (int)(ch - 'a') + 26;         /* 26..51 */
    if (ch >= '0' && ch <= '9') return (int)(ch - '0') + 52;         /* 52..61 */
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

/* Get next non-whitespace byte; returns -1 on end-of-buffer. */
static inline int next_non_ws(const char *src, size_t src_len, size_t *idx) {
    while (*idx < src_len) {
        unsigned char ch = (unsigned char)src[(*idx)++];
        if (!isspace(ch)) return (int)ch;
    }
    return -1;
}

/* Secure Base64 decode: uses src_len for bounds, validates input strictly,
   and allocates sufficient buffer with overflow checks.
   Returns NUL-terminated buffer for compatibility (binary data may contain '\0').
   On error, returns NULL. */
char *base64dec(const char *src, size_t src_len) {
    if (src == NULL) {
        fprintf(stderr, "Invalid argument: src is NULL\n");
        return NULL;
    }

    /* Compute worst-case output size: 3 * ceil(src_len/4) + 1, with overflow checks. */
    if (src_len > SIZE_MAX - 3) {
        fprintf(stderr, "Input too large\n");
        return NULL;
    }
    size_t groups_est = (src_len + 3u) / 4u;
    if (groups_est > (SIZE_MAX - 1u) / 3u) {
        fprintf(stderr, "Input too large\n");
        return NULL;
    }
    size_t out_cap = groups_est * 3u + 1u;

    char *result = (char *)malloc(out_cap);
    if (!result) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    size_t out = 0;
    size_t i = 0;

    for (;;) {
        int c1 = next_non_ws(src, src_len, &i);
        if (c1 == -1) {
            /* No data left; done. */
            break;
        }
        int c2 = next_non_ws(src, src_len, &i);
        if (c2 == -1) { fprintf(stderr, "Invalid input: truncated Base64\n"); goto fail; }

        /* Padding cannot appear in the first two positions */
        if (c1 == '=' || c2 == '=') { fprintf(stderr, "Invalid input: bad padding\n"); goto fail; }

        int v1 = b64_val((unsigned char)c1);
        int v2 = b64_val((unsigned char)c2);
        if (v1 < 0 || v2 < 0) { fprintf(stderr, "Invalid input: non-Base64 character\n"); goto fail; }

        int c3 = next_non_ws(src, src_len, &i);
        if (c3 == -1) { fprintf(stderr, "Invalid input: truncated Base64\n"); goto fail; }

        if (c3 == '=') {
            /* If third is '=', fourth must be '=' and we output 1 byte. */
            int c4 = next_non_ws(src, src_len, &i);
            if (c4 == -1) { fprintf(stderr, "Invalid input: truncated Base64\n"); goto fail; }
            if (c4 != '=') { fprintf(stderr, "Invalid input: bad padding\n"); goto fail; }

            if (out + 1 > out_cap - 1) { fprintf(stderr, "Internal error: output overflow\n"); goto fail; }
            result[out++] = (char)((v1 << 2) | (v2 >> 4));

            /* Only whitespace allowed after padding */
            while (next_non_ws(src, src_len, &i) != -1) {
                fprintf(stderr, "Invalid input: data after padding\n");
                goto fail;
            }
            break;
        }

        int v3 = b64_val((unsigned char)c3);
        if (v3 < 0) { fprintf(stderr, "Invalid input: non-Base64 character\n"); goto fail; }

        int c4 = next_non_ws(src, src_len, &i);
        if (c4 == -1) { fprintf(stderr, "Invalid input: truncated Base64\n"); goto fail; }

        if (c4 == '=') {
            /* Two output bytes; then only whitespace permitted */
            if (out + 2 > out_cap - 1) { fprintf(stderr, "Internal error: output overflow\n"); goto fail; }
            result[out++] = (char)((v1 << 2) | (v2 >> 4));
            result[out++] = (char)(((v2 & 0x0F) << 4) | (v3 >> 2));

            while (next_non_ws(src, src_len, &i) != -1) {
                fprintf(stderr, "Invalid input: data after padding\n");
                goto fail;
            }
            break;
        } else {
            int v4 = b64_val((unsigned char)c4);
            if (v4 < 0) { fprintf(stderr, "Invalid input: non-Base64 character\n"); goto fail; }

            if (out + 3 > out_cap - 1) { fprintf(stderr, "Internal error: output overflow\n"); goto fail; }
            result[out++] = (char)((v1 << 2) | (v2 >> 4));
            result[out++] = (char)(((v2 & 0x0F) << 4) | (v3 >> 2));
            result[out++] = (char)(((v3 & 0x03) << 6) | v4);
        }
    }

    /* NUL-terminate for convenience; decoded data may contain NUL bytes. */
    result[out] = '\0';
    return result;

fail:
    free(result);
    return NULL;
}
```

Notes
- If you can change the API, prefer returning the decoded length via an out parameter (and not relying on NUL-termination), which avoids ambiguity with binary data. This is a design improvement, not a security fix.
- If you need to accept URL-safe Base64, extend b64_val to return 62 for '-' and 63 for '_'.