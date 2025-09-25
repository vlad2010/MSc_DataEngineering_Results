#include <stdio.h>
#include <string.h>

#define MIN_LINE_LENGTH 80
#define MAX_LINE_LENGTH 1000

int get_line(char line[], int max) {
    int c, i;
    
    if (line == NULL || max <= 0) {
        return -1;  // Invalid parameters
    }
    
    for (i = 0; i < max - 1 && (c = getchar()) != EOF && c != '\n'; i++) {
        line[i] = c;
    }
    
    // Check if we have space for newline character before adding it
    if (c == '\n' && i < max - 1) {
        line[i] = c;
        ++i;
    }
    
    line[i] = '\0';  // Ensure the line is null-terminated
    
    // If we stopped reading because buffer is full but input continues,
    // consume the rest of the line to avoid leaving partial data in stdin
    if (i == max - 1 && c != '\n' && c != EOF) {
        int temp;
        while ((temp = getchar()) != '\n' && temp != EOF) {
            // Discard characters that don't fit in buffer
        }
    }
    
    return i;
}

int main() {
    printf("=== This program will print all the lines that exceed 80 characters, "
           "exit by CTRL+D ===\n");
    
    char line[MAX_LINE_LENGTH];
    int line_length;
    
    while ((line_length = get_line(line, MAX_LINE_LENGTH)) > 0) {
        // Changed to >= to include lines with exactly 80 characters
        if (line_length >= MIN_LINE_LENGTH) {
            // Use fputs or printf with format string to avoid format string vulnerabilities
            fputs(line, stdout);
            
            // If line was truncated (doesn't end with newline), add one
            if (line_length > 0 && line[line_length - 1] != '\n') {
                putchar('\n');
            }
        }
    }
    
    return 0;
}