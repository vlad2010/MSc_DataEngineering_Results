#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

/* Decode a single base64 character */
char base64dec_getc(const char **src) {
    while (**src && !isprint((unsigned char)**src))
        (*src)++;
    return **src ? *((*src)++) : '=';
}

/* Decode base64 encoded text */
char *base64dec(const char *src, size_t src_len) {
    char *result, *dst;
    int i;
    size_t base64_chars = 0, padding = 0;
    const char *p = src;

    // Corrected base64_digits: all entries initialized to -1
    static const signed char base64_digits[256] = {
        [0 ... 255] = -1,
        ['+'] = 62, ['/'] = 63,
        ['0'] = 52, ['1'] = 53, ['2'] = 54, ['3'] = 55, ['4'] = 56,
        ['5'] = 57, ['6'] = 58, ['7'] = 59, ['8'] = 60, ['9'] = 61,
        ['A'] = 0, ['B'] = 1, ['C'] = 2, ['D'] = 3, ['E'] = 4,
        ['F'] = 5, ['G'] = 6, ['H'] = 7, ['I'] = 8, ['J'] = 9,
        ['K'] = 10, ['L'] = 11, ['M'] = 12, ['N'] = 13, ['O'] = 14,
        ['P'] = 15, ['Q'] = 16, ['R'] = 17, ['S'] = 18, ['T'] = 19,
        ['U'] = 20, ['V'] = 21, ['W'] = 22, ['X'] = 23, ['Y'] = 24,
        ['Z'] = 25, ['a'] = 26, ['b'] = 27, ['c'] = 28, ['d'] = 29,
        ['e'] = 30, ['f'] = 31, ['g'] = 32, ['h'] = 33, ['i'] = 34,
        ['j'] = 35, ['k'] = 36, ['l'] = 37, ['m'] = 38, ['n'] = 39,
        ['o'] = 40, ['p'] = 41, ['q'] = 42, ['r'] = 43, ['s'] = 44,
        ['t'] = 45, ['u'] = 46, ['v'] = 47, ['w'] = 48, ['x'] = 49,
        ['y'] = 50, ['z'] = 51
    };

    // Count valid base64 characters and padding
    for (i = 0; i < src_len; ++i) {
        unsigned char ch = (unsigned char)src[i];
        if (ch == '=') {
            padding++;
            base64_chars++;
        } else if (base64_digits[ch] != -1) {
            base64_chars++;
        } else if (isspace(ch)) {
            continue;
        } else {
            fprintf(stderr, "Invalid character in input\n");
            return NULL;
        }
    }

    // base64_chars must be a multiple of 4
    if (base64_chars % 4 != 0) {
        fprintf(stderr, "Invalid base64 input length\n");
        return NULL;
    }

    // Calculate output length, check for overflow
    size_t out_len = (base64_chars / 4) * 3;
    if (padding > 2) {
        fprintf(stderr, "Too much padding\n");
        return NULL;
    }
    if (padding) out_len -= padding;

    if (out_len > SIZE_MAX - 1) { // check for overflow
        fprintf(stderr, "Output length overflow\n");
        return NULL;
    }

    result = dst = malloc(out_len + 1);
    if (!result) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Decode base64 characters
    size_t processed = 0;
    while (processed < base64_chars) {
        int vals[4];
        int j;
        for (j = 0; j < 4; ++j) {
            // Skip whitespace
            while (*src && isspace((unsigned char)*src)) src++;
            if (!*src) {
                vals[j] = 0;
            } else if (*src == '=') {
                vals[j] = 0;
                src++;
            } else {
                int v = base64_digits[(unsigned char)*src];
                if (v == -1) {
                    free(result);
                    fprintf(stderr, "Invalid character in input\n");
                    return NULL;
                }
                vals[j] = v;
                src++;
            }
            processed++;
        }
        *dst++ = (vals[0] << 2) | ((vals[1] & 0x30) >> 4);
        if (*(src - 2) != '=') // third char
            *dst++ = ((vals[1] & 0x0f) << 4) | ((vals[2] & 0x3c) >> 2);
        if (*(src - 1) != '=') // fourth char
            *dst++ = ((vals[2] & 0x03) << 6) | vals[3];
    }
    *dst = '\0';
    return result;
}