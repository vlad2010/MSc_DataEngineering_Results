#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>  // FIX: Added missing header for strchr

#define MAX_SIZE 100

typedef struct {
    char stack[MAX_SIZE];
    int top;
} Stack;

void initialize(Stack* stack) {
    if (stack == NULL) return;  // FIX: Added NULL check
    stack->top = -1;
}

bool is_empty(Stack* stack) {
    if (stack == NULL) return true;  // FIX: Added NULL check
    return stack->top == -1;
}

bool is_full(Stack* stack) {
    if (stack == NULL) return true;  // FIX: Added NULL check
    return stack->top == MAX_SIZE - 1;
}

// FIX: Changed return type to bool to indicate success/failure
bool push(Stack* stack, char symbol) {
    if (stack == NULL) return false;  // FIX: Added NULL check
    if (is_full(stack)) {
        // FIX: Return false instead of exit(1)
        return false;
    }
    stack->stack[++stack->top] = symbol;
    return true;
}

// FIX: Added success parameter to avoid exit on error
char pop(Stack* stack, bool* success) {
    if (success) *success = true;
    
    if (stack == NULL || is_empty(stack)) {
        if (success) *success = false;
        return '\0';  // FIX: Return null character on error instead of exit(1)
    }
    return stack->stack[stack->top--];
}

bool is_balanced(const char* expression) {
    // FIX: Added NULL check for input validation (CWE-20)
    if (expression == NULL) {
        return false;
    }
    
    Stack stack;
    initialize(&stack);
    const char* opening_symbols = "([{";
    const char* closing_symbols = ")]}";

    for (int i = 0; expression[i] != '\0'; i++) {
        char symbol = expression[i];
        
        // Check if it's an opening symbol
        if (strchr(opening_symbols, symbol) != NULL) {
            // FIX: Check push return value
            if (!push(&stack, symbol)) {
                return false;  // Stack overflow - expression too complex
            }
        } 
        // Check if it's a closing symbol
        else if (strchr(closing_symbols, symbol) != NULL) {
            if (is_empty(&stack)) {
                return false;  // There's no corresponding opening symbol
            }
            
            bool pop_success;
            char opening_symbol = pop(&stack, &pop_success);
            
            // FIX: Check if pop was successful
            if (!pop_success) {
                return false;
            }
            
            // FIX: Added NULL checks for strchr return values (CWE-476)
            const char* matching_opening_pos = strchr(opening_symbols, opening_symbol);
            const char* matching_closing_pos = strchr(closing_symbols, symbol);
            
            // This should not happen with valid stack content, but check anyway
            if (matching_opening_pos == NULL || matching_closing_pos == NULL) {
                return false;
            }
            
            // Calculate indices safely
            ptrdiff_t opening_index = matching_opening_pos - opening_symbols;
            ptrdiff_t closing_index = matching_closing_pos - closing_symbols;
            
            if (opening_index != closing_index) {
                return false;  // Opening and closing symbols don't match
            }
        }
        // Ignore other characters
    }

    return is_empty(&stack);  // If stack is empty, expression is balanced
}

int main() {
    // Test cases
    printf("%d\n", is_balanced("()"));        // Output: 1 (true)
    printf("%d\n", is_balanced("({[]})"));    // Output: 1 (true)
    printf("%d\n", is_balanced("({)}"));      // Output: 0 (false)
    printf("%d\n", is_balanced("[[[]]]"));    // Output: 1 (true)
    printf("%d\n", is_balanced("(]"));        // Output: 0 (false)
    
    // FIX: Added test for NULL input
    printf("%d\n", is_balanced(NULL));        // Output: 0 (false)
    
    return 0;
}