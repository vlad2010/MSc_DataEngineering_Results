#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int add_overflow_size(size_t a, size_t b, size_t *out) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_add_overflow)
    return __builtin_add_overflow(a, b, out);
#  endif
#endif
    // Portable fallback
    if (SIZE_MAX - a < b) return 1;
    *out = a + b;
    return 0;
}

char* format_with_commas(int num) {
    // Convert number to string safely
    char buffer[50];
    int rc = snprintf(buffer, sizeof buffer, "%d", num);
    if (rc < 0 || (size_t)rc >= sizeof buffer) {
        // Encoding error or truncation (shouldn't happen for int, but be safe)
        return NULL;
    }

    // Separate sign from digits for robust handling
    int is_negative = (buffer[0] == '-');
    const char *digits = buffer + (is_negative ? 1 : 0);
    size_t len_digits = strlen(digits);

    // Compute number of commas: one every 3 digits, except before the first group
    size_t comma_count = (len_digits > 0) ? (len_digits - 1) / 3 : 0;

    // final_len = sign + digits + commas + NUL
    size_t final_len_no_nul;
    if (add_overflow_size((size_t)is_negative, len_digits, &final_len_no_nul) ||
        add_overflow_size(final_len_no_nul, comma_count, &final_len_no_nul)) {
        return NULL; // overflow guard
    }

    size_t final_len;
    if (add_overflow_size(final_len_no_nul, (size_t)1, &final_len)) {
        return NULL; // overflow guard (for NUL)
    }

    char *result = (char *)malloc(final_len);
    if (!result) {
        return NULL;
    }

    // Fill from the end
    size_t ri = final_len - 1;
    result[ri--] = '\0';

    size_t count = 0; // digit counter for grouping
    size_t di = len_digits;
    while (di > 0) {
        result[ri--] = digits[--di];
        count++;
        if (di > 0 && count == 3) {
            result[ri--] = ',';
            count = 0;
        }
    }

    if (is_negative) {
        result[0] = '-';
    }

    return result;
}

int main(void) {
    int nums[] = {7000000, -1000, 0, INT_MIN, INT_MAX};
    size_t n = sizeof(nums)/sizeof(nums[0]);

    for (size_t k = 0; k < n; ++k) {
        char *formattedNum = format_with_commas(nums[k]);
        if (formattedNum) {
            printf("%s\n", formattedNum);
            free(formattedNum);
        } else {
            printf("Formatting failed.\n");
        }
    }

    return 0;
}