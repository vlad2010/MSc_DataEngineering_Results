#include "lexer.h"

Lexer* create_lexer(const char* source) {
    Lexer* lexer = (Lexer*)malloc(sizeof(Lexer));
    lexer->source = source;
    lexer->length = strlen(source);
    lexer->position = 0;
    return lexer;
}

void destroy_lexer(Lexer* lexer) {
    free(lexer);
}

static char peek_char(Lexer* lexer) {
    if (lexer->position >= lexer->length) {
        return '\0';
    }
    return lexer->source[lexer->position];
}

static char advance_char(Lexer* lexer) {
    if (lexer->position >= lexer->length) {
        return '\0';
    }
    return lexer->source[lexer->position++];
}

static Token make_token(TokenType type, const char* value) {
    Token token;
    token.type = type;
    token.value = strdup(value);
    return token;
}

static void skip_whitespace(Lexer* lexer) {
    while (isspace(peek_char(lexer))) {
        advance_char(lexer);
    }
}

static Token scan_number(Lexer* lexer) {
    size_t start = lexer->position - 1;
    while (isdigit(peek_char(lexer))) {
        advance_char(lexer);
    }
    char* value = strndup(lexer->source + start, lexer->position - start);
    return make_token(TOKEN_INT, value);
}

static Token scan_identifier(Lexer* lexer) {
    size_t start = lexer->position - 1;
    while (isalnum(peek_char(lexer)) || peek_char(lexer) == '_') {
        advance_char(lexer);
    }
    char* value = strndup(lexer->source + start, lexer->position - start);
    return make_token(TOKEN_IDENTIFIER, value);
}

Token next_token(Lexer* lexer) {
    skip_whitespace(lexer);
    char current = advance_char(lexer);

    if (isdigit(current)) {
        return scan_number(lexer);
    }

    if (isalpha(current) || current == '_') {
        return scan_identifier(lexer);
    }

    switch (current) {
        case '\0': return make_token(TOKEN_EOF, "");
        case '(': return make_token(TOKEN_LPAREN, "(");
        case ')': return make_token(TOKEN_RPAREN, ")");
        case ':': return make_token(TOKEN_COLON, ":");
        case ',': return make_token(TOKEN_COMMA, ",");
        case '-':
            if (peek_char(lexer) == '>') {
                advance_char(lexer);
                return make_token(TOKEN_RANGE, "->");
            }
            break;
    }

    char unknown[2] = { current, '\0' };
    return make_token(TOKEN_UNKNOWN, unknown);
}

void destroy_token(Token token) {
    free(token.value);
}