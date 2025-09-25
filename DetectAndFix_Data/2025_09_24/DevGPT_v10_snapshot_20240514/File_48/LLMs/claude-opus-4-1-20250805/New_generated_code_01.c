#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

char* format_with_commas(int num) {
    // Buffer large enough for INT_MIN with null terminator
    // INT_MIN = -2147483648 (11 chars + 1 for null)
    char buffer[20];
    int isNegative = (num < 0);
    
    // Use snprintf for safety
    snprintf(buffer, sizeof(buffer), "%d", num);
    
    // Determine the length of the number (excluding minus sign for comma calculation)
    int totalLength = strlen(buffer);
    int digitLength = isNegative ? (totalLength - 1) : totalLength;
    
    // Calculate number of commas needed (no comma if 3 or fewer digits)
    int commaCount = (digitLength > 0) ? (digitLength - 1) / 3 : 0;
    int finalLength = totalLength + commaCount;
    
    // Allocate memory for the final string
    char *result = malloc(finalLength + 1);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }
    
    // Initialize result buffer
    memset(result, 0, finalLength + 1);
    
    // Start from the end of both strings
    int srcIndex = totalLength - 1;  // Last character in source buffer
    int destIndex = finalLength - 1;  // Last position in result buffer
    
    // Copy characters from buffer to result, adding commas
    int digitCount = 0;
    
    while (srcIndex >= 0 && destIndex >= 0) {
        // Copy the current character
        result[destIndex] = buffer[srcIndex];
        
        // If it's a digit, increment digit count
        if (buffer[srcIndex] >= '0' && buffer[srcIndex] <= '9') {
            digitCount++;
            
            // Add comma if we've copied 3 digits and there are more digits to come
            if (digitCount % 3 == 0 && srcIndex > 0) {
                // Check if next character is a digit (not minus sign)
                if (buffer[srcIndex - 1] >= '0' && buffer[srcIndex - 1] <= '9') {
                    destIndex--;
                    if (destIndex >= 0) {
                        result[destIndex] = ',';
                    }
                }
            }
        }
        
        srcIndex--;
        destIndex--;
    }
    
    // Null terminate the string
    result[finalLength] = '\0';
    
    return result;
}

int main() {
    // Test cases including edge cases
    int testCases[] = {7000000, -7000000, 0, 1, -1, 1000, -1000, 
                       999, -999, INT_MAX, INT_MIN};
    int numTests = sizeof(testCases) / sizeof(testCases[0]);
    
    for (int i = 0; i < numTests; i++) {
        char *formattedNum = format_with_commas(testCases[i]);
        
        if (formattedNum) {
            printf("%d -> %s\n", testCases[i], formattedNum);
            free(formattedNum);
        } else {
            printf("Memory allocation failed for %d\n", testCases[i]);
        }
    }
    
    return 0;
}