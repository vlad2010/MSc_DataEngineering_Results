#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "parser.h"

#define MAX_SOURCE_LENGTH 1024
#define MAX_NAME_LENGTH 256

int main() {
    int ret_code = 0;
    Parser* parser = NULL;
    ConstrainedIntType* int_type = NULL;
    
    // Input validation - ensure source code is not too long
    const char* source_code = "ConstrainedInt(0->100)";
    if (source_code == NULL) {
        fprintf(stderr, "Error: source_code is NULL\n");
        return 1;
    }
    
    size_t source_len = strnlen(source_code, MAX_SOURCE_LENGTH + 1);
    if (source_len > MAX_SOURCE_LENGTH) {
        fprintf(stderr, "Error: source_code exceeds maximum length\n");
        return 1;
    }
    
    // Create parser with NULL check
    parser = create_parser(source_code);
    if (parser == NULL) {
        fprintf(stderr, "Error: Failed to create parser\n");
        return 1;
    }
    
    // Parse with NULL check
    int_type = parse_constrained_int_type(parser);
    if (int_type == NULL) {
        fprintf(stderr, "Error: Failed to parse constrained int type\n");
        ret_code = 1;
        goto cleanup;
    }
    
    // Validate parsed data before use
    if (int_type->name == NULL) {
        fprintf(stderr, "Error: Parsed type has NULL name\n");
        ret_code = 1;
        goto cleanup;
    }
    
    // Validate name length to prevent buffer issues
    size_t name_len = strnlen(int_type->name, MAX_NAME_LENGTH + 1);
    if (name_len > MAX_NAME_LENGTH) {
        fprintf(stderr, "Error: Type name exceeds maximum length\n");
        ret_code = 1;
        goto cleanup;
    }
    
    // Validate range values
    if (int_type->min_value > int_type->max_value) {
        fprintf(stderr, "Error: Invalid range (min > max)\n");
        ret_code = 1;
        goto cleanup;
    }
    
    // Safe output with controlled format strings
    printf("Parsed constrained int type:\n");
    printf("Name: %.*s\n", MAX_NAME_LENGTH, int_type->name);
    printf("Range: %d -> %d\n", int_type->min_value, int_type->max_value);
    
cleanup:
    // Safe cleanup with NULL checks
    // Assuming destroy_parser handles NULL gracefully and frees all parser-related resources
    if (parser != NULL) {
        destroy_parser(parser);
        parser = NULL;
    }
    
    // Only free int_type structure if it was allocated
    // Assuming the name is freed by destroy_parser or is part of int_type allocation
    // If not, uncomment the following:
    /*
    if (int_type != NULL) {
        if (int_type->name != NULL) {
            free(int_type->name);
            int_type->name = NULL;
        }
        free(int_type);
        int_type = NULL;
    }
    */
    
    // Alternative: If int_type needs separate cleanup and name is separately allocated:
    if (int_type != NULL) {
        // Only free name if it's a separate allocation
        // This depends on the implementation of parse_constrained_int_type
        // Comment out if name is freed by destroy_parser or is static/stack allocated
        if (int_type->name != NULL) {
            free(int_type->name);
            int_type->name = NULL;
        }
        free(int_type);
        int_type = NULL;
    }
    
    return ret_code;
}