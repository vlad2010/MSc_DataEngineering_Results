#ifndef LEXER_H
#define LEXER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    TOKEN_INT,
    TOKEN_RANGE,
    TOKEN_IDENTIFIER,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_COLON,
    TOKEN_COMMA,
    TOKEN_EOF,
    TOKEN_UNKNOWN
} TokenType;

typedef struct {
    TokenType type;
    char* value;
} Token;

typedef struct {
    const char* source;
    size_t length;
    size_t position;
} Lexer;

Lexer* create_lexer(const char* source);
void destroy_lexer(Lexer* lexer);
Token next_token(Lexer* lexer);
void destroy_token(Token token);

#endif // LEXER_H