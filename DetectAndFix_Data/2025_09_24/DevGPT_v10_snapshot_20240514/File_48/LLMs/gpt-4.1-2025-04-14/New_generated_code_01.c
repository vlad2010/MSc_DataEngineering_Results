#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

char* format_with_commas(int num) {
    // Buffer for the initial conversion without commas
    char buffer[50];
    // Use snprintf to avoid buffer overflow (CWE-120, CWE-676)
    int written = snprintf(buffer, sizeof(buffer), "%d", num);
    if (written < 0 || written >= (int)sizeof(buffer)) {
        // Encoding error or buffer too small
        return NULL;
    }

    int length = strlen(buffer);
    int isNegative = (buffer[0] == '-');
    int digits = isNegative ? length - 1 : length;

    // Calculate number of commas needed
    int commaCount = (digits - 1) / 3;
    int finalLength = length + commaCount;

    // Check for integer overflow (CWE-190)
    if (finalLength < length || finalLength < 0) {
        return NULL;
    }

    // Allocate memory for the final string (+1 for null terminator)
    char *result = malloc(finalLength + 1);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }

    int i = length - 1;
    int j = finalLength;
    result[j--] = '\0'; // Null-terminate

    int commaPosition = 0;
    while (i >= 0) {
        if (isNegative && i == 0) {
            result[j--] = '-';
            break;
        }
        result[j--] = buffer[i--];
        commaPosition++;
        if (commaPosition == 3 && i >= (isNegative ? 1 : 0)) {
            result[j--] = ',';
            commaPosition = 0;
        }
    }

    return result;
}

int main() {
    int num = 7000000;
    char *formattedNum = format_with_commas(num);

    if (formattedNum) {
        printf("%s\n", formattedNum);
        free(formattedNum); // Don't forget to free the allocated memory
    } else {
        printf("Formatting failed or memory allocation failed.\n");
    }

    return 0;
}