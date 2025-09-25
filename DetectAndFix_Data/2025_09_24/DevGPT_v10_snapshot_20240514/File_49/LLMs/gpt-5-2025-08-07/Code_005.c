#include <stdio.h>
#include "parser.h"

int main() {
    const char* source_code = "ConstrainedInt(0->100)";
    Parser* parser = create_parser(source_code);

    ConstrainedIntType* int_type = parse_constrained_int_type(parser);
    printf("Parsed constrained int type:\n");
    printf("Name: %s\n", int_type->name);
    printf("Range: %d -> %d\n", int_type->min_value, int_type->max_value);

    destroy_parser(parser);
    free(int_type->name);
    free(int_type);

    return 0;
}