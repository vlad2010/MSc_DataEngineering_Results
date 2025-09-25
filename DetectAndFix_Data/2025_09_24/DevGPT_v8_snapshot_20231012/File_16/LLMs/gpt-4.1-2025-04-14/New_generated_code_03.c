#include <stdio.h>

// Example function to log a user message safely
void log_message(const char *user_message) {
    // Never use user_message as the format string!
    // Use "%s" as the format string to avoid CWE-134
    fprintf(stderr, "Log: %s\n", user_message);
}

int main() {
    const char *user_input = "This is a user message";
    log_message(user_input);
    return 0;
}