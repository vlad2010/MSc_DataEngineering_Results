#include <stdio.h>
#include <errno.h>

/*
 * Securely write a message to stderr:
 * - Treats msg as data (no format-string risk)
 * - Neutralizes control characters (prevents log forging and terminal escape injection)
 * - Uses flockfile/funlockfile to reduce interleaving in multithreaded contexts
 * - Checks for write errors
 *
 * NOTES:
 * - Do not pass secrets here (tokens, passwords, keys, PII).
 * - Consider routing logs to a proper logging framework for rotation/permissions.
 */
static void secure_log_stderr(const char *msg) {
    if (msg == NULL) {
        msg = "(null)";
    }

    flockfile(stderr);  // reduce interleaving from multiple threads

    // Sanitize and emit
    const unsigned char *p = (const unsigned char *)msg;
    for (; *p; ++p) {
        unsigned char c = *p;
        if (c == '\n') {
            if (fputs("\\n", stderr) == EOF) goto write_error;
        } else if (c == '\r') {
            if (fputs("\\r", stderr) == EOF) goto write_error;
        } else if (c == '\t') {
            if (fputs("\\t", stderr) == EOF) goto write_error;
        } else if (c == 0x1B) { // ESC
            if (fputs("\\e", stderr) == EOF) goto write_error;
        } else if (c < 0x20 || c == 0x7F) { // other control chars
            if (fprintf(stderr, "\\x%02X", c) < 0) goto write_error;
        } else {
            if (fputc(c, stderr) == EOF) goto write_error;
        }
    }
    if (fputc('\n', stderr) == EOF) goto write_error;

    // Flush to detect errors promptly (stderr is usually unbuffered, but be explicit)
    if (fflush(stderr) == EOF) goto write_error;

    funlockfile(stderr);
    return;

write_error:
    // Best-effort unlock; avoid recursive logging to stderr on failure
    funlockfile(stderr);
    // In real applications, consider reporting via syslog or another channel.
    // Here we just suppress further output to avoid looping on errors.
}

/* Example usage */
int main(void) {
    // Original safe constant message
    secure_log_stderr("Your log message here");

    // Example with untrusted/user input (demonstrates neutralization)
    const char *user_supplied = "User said: hello\nand tried to inject \x1B[31mRED\x1B[0m";
    secure_log_stderr(user_supplied);

    // NEVER do this with untrusted data (CWE-134):
    // fprintf(stderr, user_supplied);

    // If you truly need formatting, lock the format string and only interpolate data:
    // const char *fmt = "Result: %d";
    // fprintf(stderr, fmt, 42);

    return 0;
}