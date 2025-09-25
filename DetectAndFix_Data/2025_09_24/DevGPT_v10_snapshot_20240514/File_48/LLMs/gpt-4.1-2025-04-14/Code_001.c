#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* format_with_commas(int num) {
    // Buffer for the initial conversion without commas
    char buffer[50];
    sprintf(buffer, "%d", num);

    // Determine the length of the final string
    int length = strlen(buffer);
    int commaCount = (length - 1) / 3;
    int finalLength = length + commaCount;

    // Allocate memory for the final string
    char *result = malloc(finalLength + 1);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }

    int i = length - 1, j = finalLength;
    result[j--] = '\0'; // Null-terminating the string

    // Copy characters from buffer to result, adding commas
    int commaPosition = 3;
    while (i >= 0) {
        result[j--] = buffer[i--];
        commaPosition--;
        if (commaPosition == 0 && j > 0) {
            result[j--] = ',';
            commaPosition = 3;
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
        printf("Memory allocation failed.\n");
    }

    return 0;
}