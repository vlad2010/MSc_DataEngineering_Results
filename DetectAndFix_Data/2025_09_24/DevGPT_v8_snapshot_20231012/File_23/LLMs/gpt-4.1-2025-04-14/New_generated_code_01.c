#include <stdio.h>

#define MIN_LINE_LENGTH 80
#define MAX_LINE_LENGTH 1000

// Secure version: Pass buffer size using sizeof, not macro
int get_line(char line[], size_t max) {
    int c;
    size_t i;

    // Defensive: Ensure max > 0
    if (max == 0) {
        return 0;
    }

    for (i = 0; i < max - 1 && (c = getchar()) != EOF && c != '\n'; i++) {
        line[i] = (char)c;
    }

    if (c == '\n' && i < max - 1) {
        line[i] = (char)c;
        ++i;
    }

    line[i < max ? i : max - 1] = '\0';  // Ensure null-termination

    return (int)i;
}

int main(void) {
    printf("=== This program will print all the lines that exceed 80 characters, "
           "exit by CTRL+D ===\n");

    char line[MAX_LINE_LENGTH];
    int line_length;

    while ((line_length = get_line(line, sizeof(line))) > 0) {
        if (line_length > MIN_LINE_LENGTH) {
            // Always use a format string to avoid CWE-134
            printf("%s", line);
        }
    }

    return 0;
}