#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

/* Decode a single base64 character */
static int base64dec_getc(const char **src, const char *end) {
    if (!src || !*src || !end)
        return -1;
    
    while (*src < end && **src && !isprint((unsigned char)**src))
        (*src)++;
    
    if (*src >= end || !**src)
        return '=';
    
    return *((*src)++);
}

/* Decode base64 encoded text */
char *base64dec(const char *src, size_t src_len) {
    char *result, *dst;
    const char *src_end;
    size_t max_output_len;
    
    /* Initialize base64 decoding table with -1 for invalid characters */
    static const signed char base64_digits[256] = {
        ['A'] = 0,  ['B'] = 1,  ['C'] = 2,  ['D'] = 3,  ['E'] = 4,  ['F'] = 5,
        ['G'] = 6,  ['H'] = 7,  ['I'] = 8,  ['J'] = 9,  ['K'] = 10, ['L'] = 11,
        ['M'] = 12, ['N'] = 13, ['O'] = 14, ['P'] = 15, ['Q'] = 16, ['R'] = 17,
        ['S'] = 18, ['T'] = 19, ['U'] = 20, ['V'] = 21, ['W'] = 22, ['X'] = 23,
        ['Y'] = 24, ['Z'] = 25, ['a'] = 26, ['b'] = 27, ['c'] = 28, ['d'] = 29,
        ['e'] = 30, ['f'] = 31, ['g'] = 32, ['h'] = 33, ['i'] = 34, ['j'] = 35,
        ['k'] = 36, ['l'] = 37, ['m'] = 38, ['n'] = 39, ['o'] = 40, ['p'] = 41,
        ['q'] = 42, ['r'] = 43, ['s'] = 44, ['t'] = 45, ['u'] = 46, ['v'] = 47,
        ['w'] = 48, ['x'] = 49, ['y'] = 50, ['z'] = 51, ['0'] = 52, ['1'] = 53,
        ['2'] = 54, ['3'] = 55, ['4'] = 56, ['5'] = 57, ['6'] = 58, ['7'] = 59,
        ['8'] = 60, ['9'] = 61, ['+'] = 62, ['/'] = 63, ['='] = -2,
        /* All other elements implicitly initialized to 0 */
    };
    
    /* Validate input */
    if (!src || src_len == 0) {
        fprintf(stderr, "Invalid input\n");
        return NULL;
    }
    
    /* Check for integer overflow and calculate safe allocation size */
    if (src_len > SIZE_MAX / 3) {
        fprintf(stderr, "Input too large\n");
        return NULL;
    }
    max_output_len = ((src_len + 3) / 4) * 3 + 1;
    
    /* Allocate memory for decoded string */
    result = dst = calloc(1, max_output_len);
    if (!result) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    
    src_end = src + src_len;
    size_t output_count = 0;
    
    /* Decode base64 characters */
    while (src < src_end && *src) {
        int chars[4];
        int values[4];
        int i;
        
        /* Read 4 base64 characters */
        for (i = 0; i < 4; i++) {
            chars[i] = base64dec_getc(&src, src_end);
            if (chars[i] == -1) {
                chars[i] = '=';
            }
            
            /* Look up value, treating anything invalid as padding */
            if (chars[i] == '=') {
                values[i] = -2;
            } else {
                int idx = (unsigned char)chars[i];
                values[i] = base64_digits[idx];
                if (values[i] == 0 && chars[i] != 'A') {
                    /* Invalid character */
                    fprintf(stderr, "Invalid base64 character: %c\n", chars[i]);
                    free(result);
                    return NULL;
                }
            }
        }
        
        /* Validate first two characters must be valid base64 */
        if (values[0] < 0 || values[1] < 0) {
            fprintf(stderr, "Invalid base64 sequence\n");
            free(result);
            return NULL;
        }
        
        /* Bounds check before writing */
        if (output_count >= max_output_len - 3) {
            fprintf(stderr, "Output buffer overflow prevented\n");
            free(result);
            return NULL;
        }
        
        /* Decode and write bytes */
        *dst++ = (values[0] << 2) | ((values[1] & 0x30) >> 4);
        output_count++;
        
        if (values[2] == -2) /* Padding */
            break;
        if (values[2] < 0) {
            fprintf(stderr, "Invalid base64 character in position 3\n");
            free(result);
            return NULL;
        }
        
        *dst++ = ((values[1] & 0x0f) << 4) | ((values[2] & 0x3c) >> 2);
        output_count++;
        
        if (values[3] == -2) /* Padding */
            break;
        if (values[3] < 0) {
            fprintf(stderr, "Invalid base64 character in position 4\n");
            free(result);
            return NULL;
        }
        
        *dst++ = ((values[2] & 0x03) << 6) | values[3];
        output_count++;
    }
    
    *dst = '\0';
    return result;
}