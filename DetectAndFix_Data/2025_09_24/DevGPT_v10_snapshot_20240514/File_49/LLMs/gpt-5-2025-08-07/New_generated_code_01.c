#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "parser.h"

/*
Assumptions for safe cleanup:
- The parser library provides a destructor for the parsed type:
    void destroy_constrained_int_type(ConstrainedIntType* t);
  If your library does not provide this, add one there. Do NOT free nested
  fields (e.g., name) directly here unless the API explicitly documents that
  the caller owns them. This avoids CWE-415 (double free) and lifetime bugs.
*/

static void print_constrained_int_type(const ConstrainedIntType* t) {
    if (!t) {
        fputs("No type to print.\n", stderr);
        return;
    }
    if (!t->name) {
        fputs("Parsed type has NULL name.\n", stderr);
        return;
    }

    // Use wide cast to avoid printf format/type mismatch (CWE-686)
    long long minv = (long long)t->min_value;
    long long maxv = (long long)t->max_value;

    printf("Parsed constrained int type:\n");
    printf("Name: %s\n", t->name);
    printf("Range: %lld -> %lld\n", minv, maxv);

    // Basic validation (CWE-20)
    if (minv > maxv) {
        fprintf(stderr, "Warning: invalid range (min > max): %lld > %lld\n", minv, maxv);
    }
}

int main(void) {
    const char* source_code = "ConstrainedInt(0->100)";
    Parser* parser = NULL;
    ConstrainedIntType* int_type = NULL;

    parser = create_parser(source_code);
    if (!parser) {
        fputs("Error: create_parser failed.\n", stderr);
        return EXIT_FAILURE;
    }

    int_type = parse_constrained_int_type(parser);
    if (!int_type) {
        fputs("Error: parse_constrained_int_type failed.\n", stderr);
        destroy_parser(parser);
        return EXIT_FAILURE;
    }

    if (!int_type->name) {
        fputs("Error: parsed type has NULL name.\n", stderr);
        destroy_constrained_int_type(int_type);  // library-owned cleanup
        destroy_parser(parser);
        return EXIT_FAILURE;
    }

    print_constrained_int_type(int_type);

    // Cleanup: destroy parsed object before parser (lifetime/ownership safe)
    destroy_constrained_int_type(int_type);
    destroy_parser(parser);

    return EXIT_SUCCESS;
}