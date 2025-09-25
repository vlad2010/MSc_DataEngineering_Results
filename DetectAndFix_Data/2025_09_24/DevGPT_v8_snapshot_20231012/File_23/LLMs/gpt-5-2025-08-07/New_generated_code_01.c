#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <stddef.h>

#define MIN_LINE_LENGTH 80
#define MAX_LINE_LENGTH 1000

/* Print a string safely by length, escaping control characters except '\n'.
   This mitigates terminal escape sequence injection (CWE-150). */
static void print_sanitized(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '\n') {
            putchar('\n');
        } else if (isprint(ch) || ch == '\t') {
            putchar(ch);
        } else {
            /* Escape control/non-printable bytes */
            printf("\\x%02X", ch);
        }
    }
}

/* Secure line reader:
   - Consumes the entire logical line from stdin (up to and including '\n' or EOF),
     counting its total length in *total_len.
   - Stores up to (cap - 1) bytes into buf, NUL-terminates when cap > 0.
   - Sets *had_newline if a newline ended the line.
   - Returns stored length on success; returns -1 on true EOF with no data read.
   This fixes:
     * CWE-457 (uninitialized c) by initializing c and handling small caps.
     * Truncation logic: always consume rest of line and report total length. */
static int get_line_secure(char *buf, size_t cap, size_t *total_len, bool *had_newline) {
    size_t i = 0;
    size_t total = 0;
    int c = EOF;
    bool nl = false;

    if (buf && cap > 0) {
        buf[0] = '\0';
    }

    for (;;) {
        c = getchar();
        if (c == EOF) {
            break;
        }
        total++;

        if (c == '\n') {
            nl = true;
            if (buf && cap > 0 && i < cap - 1) {
                buf[i++] = (char)c;
            }
            break;
        }

        if (buf && cap > 0 && i < cap - 1) {
            buf[i++] = (char)c;
        }
    }

    if (buf && cap > 0) {
        buf[i] = '\0';
    }
    if (total_len) {
        *total_len = total; /* includes newline if present */
    }
    if (had_newline) {
        *had_newline = nl;
    }

    /* True EOF with no data read at all */
    if (c == EOF && total == 0) {
        return -1;
    }

    return (int)i; /* number of bytes stored in buf (may be truncated) */
}

int main(void) {
    printf("=== This program will print all the lines that exceed %d characters, "
           "exit by CTRL+D ===\n", MIN_LINE_LENGTH);

    char line[MAX_LINE_LENGTH];
    size_t total_len = 0;
    bool had_newline = false;

    for (;;) {
        int stored_len = get_line_secure(line, sizeof(line), &total_len, &had_newline);
        if (stored_len < 0) {
            break; /* EOF */
        }

        /* total_len reflects the real logical line length (including newline if present) */
        if (total_len > MIN_LINE_LENGTH) {
            bool truncated = (total_len > (size_t)stored_len);

            /* Print sanitized content by exact stored length to avoid format issues */
            print_sanitized(line, (size_t)stored_len);

            /* Ensure each output ends with a newline for readability */
            if (truncated) {
                /* If truncated, we likely dropped the trailing newline; add an indicator */
                fputs(" [truncated]\n", stdout);
            } else if (!had_newline) {
                /* No newline in input; add one to keep outputs line-oriented */
                fputc('\n', stdout);
            }
        }
    }

    return 0;
}