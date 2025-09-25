#ifndef PARSER_H
#define PARSER_H

#include "lexer.h"

typedef struct {
    char* name;
    int min_value;
    int max_value;
} ConstrainedIntType;

typedef struct {
    Lexer* lexer;
    Token current_token;
} Parser;

Parser* create_parser(const char* source);
void destroy_parser(Parser* parser);
ConstrainedIntType* parse_constrained_int_type(Parser* parser);

#endif // PARSER_H