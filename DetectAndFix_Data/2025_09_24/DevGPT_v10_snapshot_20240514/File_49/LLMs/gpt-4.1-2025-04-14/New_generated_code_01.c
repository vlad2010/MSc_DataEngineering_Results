#include <stdio.h>
#include <stdlib.h>
#include "parser.h"

int main() {
    const char* source_code = "ConstrainedInt(0->100)";
    Parser* parser = create_parser(source_code);
    if (parser == NULL) {
        fprintf(stderr, "Failed to create parser\n");
        return 1;
    }

    ConstrainedIntType* int_type = parse_constrained_int_type(parser);
    if (int_type == NULL) {
        fprintf(stderr, "Failed to parse constrained int type\n");
        destroy_parser(parser);
        return 1;
    }

    printf("Parsed constrained int type:\n");
    // Use "%s" to avoid format string vulnerabilities
    printf("Name: %s\n", int_type->name ? int_type->name : "(null)");
    printf("Range: %d -> %d\n", int_type->min_value, int_type->max_value);

    destroy_parser(parser);
    // Free only if not NULL
    if (int_type->name) free(int_type->name);
    free(int_type);

    return 0;
}