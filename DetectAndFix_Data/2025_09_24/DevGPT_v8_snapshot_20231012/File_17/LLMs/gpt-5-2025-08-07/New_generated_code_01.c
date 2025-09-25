#include <syslog.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_LOG_LEN 1024

// Sanitize untrusted input for logs: escape control chars to prevent log injection (CWE-117).
static void sanitize_for_log(const char *in, char *out, size_t outsz) {
    if (!out || outsz == 0) return;
    size_t j = 0;
    if (!in) { out[0] = '\0'; return; }

    const char hex[] = "0123456789ABCDEF";
    for (size_t i = 0; in[i] != '\0' && j + 1 < outsz; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '\n') {
            if (j + 2 < outsz) { out[j++]='\\'; out[j++]='n'; }
        } else if (c == '\r') {
            if (j + 2 < outsz) { out[j++]='\\'; out[j++]='r'; }
        } else if (c == '\t') {
            if (j + 2 < outsz) { out[j++]='\\'; out[j++]='t'; }
        } else if (isprint(c)) {
            out[j++] = (char)c;
        } else {
            if (j + 4 < outsz) {
                out[j++]='\\'; out[j++]='x';
                out[j++] = hex[(c >> 4) & 0xF];
                out[j++] = hex[c & 0xF];
            } else {
                break;
            }
        }
    }
    out[j] = '\0';
}

// Safe wrapper for syslog to avoid CWE-134 and CWE-117.
static void safe_syslog_debug(const char *app_name, const char *untrusted_msg) {
    static int initialized = 0;
    if (!initialized) {
        // Use LOG_PID for traceability. Choose facility per data type:
        // - LOG_USER for general app logs
        // - LOG_AUTHPRIV for authentication-related logs (restricted permissions)
        openlog(app_name ? app_name : "myapp", LOG_PID | LOG_NDELAY, LOG_USER);

        // In production, drop DEBUG logs to reduce exposure (CWE-532) and noise.
        #ifdef NDEBUG
        setlogmask(LOG_UPTO(LOG_INFO));
        #else
        setlogmask(LOG_UPTO(LOG_DEBUG));
        #endif

        initialized = 1;
    }

    char safe[MAX_LOG_LEN];
    sanitize_for_log(untrusted_msg, safe, sizeof(safe));

    // Always use an explicit format string to prevent format-string vulns (CWE-134).
    syslog(LOG_DEBUG, "%s", safe);
}

int main(void) {
    // Example usage
    const char *tainted = "Your log message here"; // Could be user-controlled in real code
    // Important: never include secrets/PII in logs (CWE-532).
    safe_syslog_debug("example_app", tainted);

    closelog();
    return 0;
}