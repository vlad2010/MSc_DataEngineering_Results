#include "parser.h"

Parser* create_parser(const char* source) {
    Parser* parser = (Parser*)malloc(sizeof(Parser));
    parser->lexer = create_lexer(source);
    parser->current_token = next_token(parser->lexer);
    return parser;
}

void destroy_parser(Parser* parser) {
    destroy_token(parser->current_token);
    destroy_lexer(parser->lexer);
    free(parser);
}

static void consume_token(Parser* parser, TokenType expected) {
    if (parser->current_token.type == expected) {
        destroy_token(parser->current_token);
        parser->current_token = next_token(parser->lexer);
    } else {
        fprintf(stderr, "Error: Expected token of type %d but got %d\n", expected, parser->current_token.type);
        exit(1);
    }
}

static int parse_integer(Parser* parser) {
    int value = atoi(parser->current_token.value);
    consume_token(parser, TOKEN_INT);
    return value;
}

ConstrainedIntType* parse_constrained_int_type(Parser* parser) {
    ConstrainedIntType* result = (ConstrainedIntType*)malloc(sizeof(ConstrainedIntType));

    if (parser->current_token.type != TOKEN_IDENTIFIER) {
        fprintf(stderr, "Error: Expected identifier for type name\n");
        exit(1);
    }

    result->name = strdup(parser->current_token.value);
    consume_token(parser, TOKEN_IDENTIFIER);

    consume_token(parser, TOKEN_LPAREN);
    result->min_value = parse_integer(parser);
    consume_token(parser, TOKEN_RANGE);
    result->max_value = parse_integer(parser);
    consume_token(parser, TOKEN_RPAREN);

    return result;
}