#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    char *buf;
    size_t top;      // number of elements in stack
    size_t capacity; // allocated size of buf
} Stack;

static bool stack_init(Stack *s, size_t initial_capacity) {
    if (!s) return false;
    if (initial_capacity == 0) initial_capacity = 16;
    s->buf = (char *)malloc(initial_capacity * sizeof(char));
    if (!s->buf) {
        s->capacity = s->top = 0;
        return false;
    }
    s->capacity = initial_capacity;
    s->top = 0;
    return true;
}

static void stack_free(Stack *s) {
    if (!s) return;
    free(s->buf);
    s->buf = NULL;
    s->capacity = 0;
    s->top = 0;
}

static bool stack_push(Stack *s, char c) {
    if (!s) return false;
    if (s->top == s->capacity) {
        // Grow capacity by 2x, cap at SIZE_MAX/2 for safety (sizeof(char) == 1)
        size_t new_capacity = (s->capacity == 0) ? 16 : s->capacity * 2;
        if (new_capacity < s->capacity) { // overflow check
            return false;
        }
        char *new_buf = (char *)realloc(s->buf, new_capacity);
        if (!new_buf) {
            return false;
        }
        s->buf = new_buf;
        s->capacity = new_capacity;
    }
    s->buf[s->top++] = c;
    return true;
}

static bool stack_pop(Stack *s, char *out) {
    if (!s || s->top == 0) return false;
    char v = s->buf[--s->top];
    if (out) *out = v;
    return true;
}

bool is_balanced(const char* expression) {
    if (expression == NULL) {
        return false; // CWE-476 fix
    }

    const char *opening_symbols = "([{";
    const char *closing_symbols = ")]}";

    Stack stack;
    if (!stack_init(&stack, 32)) {
        // Allocation failure: fail closed without crashing
        return false;
    }

    bool ok = true;

    for (size_t i = 0; expression[i] != '\0'; i++) { // size_t avoids CWE-190
        char symbol = expression[i];

        const char *open_pos = strchr(opening_symbols, symbol);
        if (open_pos != NULL) {
            if (!stack_push(&stack, symbol)) { // CWE-703 fix: handle errors
                ok = false;
                break;
            }
            continue;
        }

        const char *close_pos = strchr(closing_symbols, symbol);
        if (close_pos != NULL) {
            char opening_symbol;
            if (!stack_pop(&stack, &opening_symbol)) { // underflow -> unbalanced
                ok = false;
                break;
            }
            // Find the indexes to ensure the bracket types match
            const char *opening_match_pos = strchr(opening_symbols, opening_symbol);
            // opening_symbol must be one of opening_symbols because we only push those
            if (opening_match_pos == NULL) {
                ok = false;
                break;
            }
            size_t open_idx = (size_t)(opening_match_pos - opening_symbols);
            size_t close_idx = (size_t)(close_pos - closing_symbols);

            if (open_idx != close_idx) {
                ok = false;
                break;
            }
        }
        // Any other characters are ignored (not a security concern)
    }

    if (ok && stack.top != 0) {
        ok = false; // Unclosed openings
    }

    stack_free(&stack);
    return ok;
}

int main(void) {
    printf("%d\n", is_balanced("()"));       // Output: 1 (true)
    printf("%d\n", is_balanced("({[]})"));   // Output: 1 (true)
    printf("%d\n", is_balanced("({)}"));     // Output: 0 (false)
    printf("%d\n", is_balanced("[[[]]]"));   // Output: 1 (true)
    printf("%d\n", is_balanced("(]"));       // Output: 0 (false)

    // Demonstrate resilience (no crash/exit) on deep nesting
    // char deep[10000]; memset(deep, '(', 9998); deep[9998] = ')'; deep[9999] = '\0';
    // printf("%d\n", is_balanced(deep));

    return 0;
}