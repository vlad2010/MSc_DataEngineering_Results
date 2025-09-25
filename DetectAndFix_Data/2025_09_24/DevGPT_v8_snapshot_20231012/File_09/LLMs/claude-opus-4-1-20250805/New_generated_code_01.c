#include <stdio.h>
#include <inttypes.h>
#include <limits.h>
#include <errno.h>
#include <stdlib.h>

int main() {
    // CWE-457 Fix: Initialize variable
    int64_t value = 0;
    int64_t *p = &value;
    int scan_result;
    
    // Provide clear input prompt
    printf("Please enter an integer value: ");
    fflush(stdout);
    
    // Clear errno before scanf to detect overflow
    errno = 0;
    
    // Use the correct format specifier and pass a pointer to int64_t
    scan_result = scanf("%" SCNd64, p);
    
    // CWE-252 Fix: Comprehensive return value checking
    if (scan_result == EOF) {
        fprintf(stderr, "Error: End of file reached or read error occurred\n");
        return EXIT_FAILURE;
    } else if (scan_result == 0) {
        fprintf(stderr, "Error: Invalid input format - expected an integer\n");
        // Clear invalid input from buffer
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
        return EXIT_FAILURE;
    } else if (scan_result == 1) {
        // CWE-20 Fix: Check for overflow conditions
        if (errno == ERANGE) {
            fprintf(stderr, "Error: Integer overflow - value out of range for int64_t\n");
            return EXIT_FAILURE;
        }
        
        // Additional validation: Check for reasonable bounds (example)
        // You can adjust these bounds based on your application requirements
        const int64_t MIN_ALLOWED = INT64_MIN;
        const int64_t MAX_ALLOWED = INT64_MAX;
        
        if (value < MIN_ALLOWED || value > MAX_ALLOWED) {
            fprintf(stderr, "Error: Value %" PRId64 " is outside acceptable range [%" PRId64 ", %" PRId64 "]\n", 
                    value, MIN_ALLOWED, MAX_ALLOWED);
            return EXIT_FAILURE;
        }
        
        // Input was successfully read and validated
        printf("Value: %" PRId64 "\n", value);
        
        // Clear any remaining input from buffer
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
        
    } else {
        fprintf(stderr, "Error: Unexpected scanf return value: %d\n", scan_result);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}