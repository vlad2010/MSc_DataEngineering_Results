#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 100

typedef struct {
    char stack[MAX_SIZE];
    int top;
} Stack;

void initialize(Stack* stack) {
    stack->top = -1;
}

bool is_empty(Stack* stack) {
    return stack->top == -1;
}

bool is_full(Stack* stack) {
    return stack->top == MAX_SIZE - 1;
}

// Return 0 on success, -1 on overflow
int push(Stack* stack, char symbol) {
    if (is_full(stack)) {
        // Stack overflow
        return -1;
    }
    stack->stack[++stack->top] = symbol;
    return 0;
}

// Return 0 on success, -1 on underflow
int pop(Stack* stack, char* symbol) {
    if (is_empty(stack)) {
        // Stack underflow
        return -1;
    }
    *symbol = stack->stack[stack->top--];
    return 0;
}

// Return 1 if balanced, 0 if not, -1 on error (e.g., stack overflow/underflow)
int is_balanced(const char* expression) {
    Stack stack;
    initialize(&stack);
    const char* opening_symbols = "([{";
    const char* closing_symbols = ")]}";

    // Input validation: limit input length to MAX_SIZE*2 (arbitrary, can be adjusted)
    size_t expr_len = strlen(expression);
    if (expr_len > MAX_SIZE * 2) {
        // Input too long
        return -1;
    }

    for (size_t i = 0; expression[i] != '\0'; i++) {
        char symbol = expression[i];
        if (strchr(opening_symbols, symbol) != NULL) {
            if (push(&stack, symbol) != 0) {
                // Stack overflow
                return -1;
            }
        } else if (strchr(closing_symbols, symbol) != NULL) {
            char opening_symbol;
            if (pop(&stack, &opening_symbol) != 0) {
                // Stack underflow
                return 0;  // Not balanced
            }
            const char* matching_opening_symbol = strchr(opening_symbols, opening_symbol);
            const char* matching_closing_symbol = strchr(closing_symbols, symbol);
            if ((matching_opening_symbol == NULL) || (matching_closing_symbol == NULL) ||
                (matching_opening_symbol - opening_symbols != matching_closing_symbol - closing_symbols)) {
                return 0;  // Not balanced
            }
        }
    }

    return is_empty(&stack) ? 1 : 0;  // 1 if balanced, 0 if not
}

int main() {
    int result;

    result = is_balanced("()");
    printf("%d\n", result);  // Output: 1 (true)

    result = is_balanced("({[]})");
    printf("%d\n", result);  // Output: 1 (true)

    result = is_balanced("({)}");
    printf("%d\n", result);  // Output: 0 (false)

    result = is_balanced("[[[]]]");
    printf("%d\n", result);  // Output: 1 (true)

    result = is_balanced("(]");
    printf("%d\n", result);  // Output: 0 (false)

    // Test for overflow
    char long_expr[MAX_SIZE * 2 + 2];
    memset(long_expr, '(', MAX_SIZE * 2 + 1);
    long_expr[MAX_SIZE * 2 + 1] = '\0';
    result = is_balanced(long_expr);
    printf("%d\n", result);  // Output: -1 (error: input too long)

    return 0;
}