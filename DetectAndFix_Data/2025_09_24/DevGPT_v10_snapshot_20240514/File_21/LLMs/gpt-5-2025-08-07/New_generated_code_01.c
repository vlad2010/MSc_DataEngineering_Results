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