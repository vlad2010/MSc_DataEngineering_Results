#include <syslog.h>
#include <stdlib.h>

// Example function to demonstrate secure logging
void log_event(const char *event_detail) {
    // Only log debug messages if DEBUG_MODE is enabled
    const char *debug_mode = getenv("DEBUG_MODE");
    if (debug_mode && strcmp(debug_mode, "1") == 0) {
        // Use a constant format string to avoid format string vulnerabilities (CWE-134)
        // Never log sensitive information (CWE-532)
        syslog(LOG_DEBUG, "Event occurred: %s", event_detail);
    } else {
        // Use a higher log level for production
        syslog(LOG_INFO, "Event occurred");
    }
}

int main() {
    // Example usage
    log_event("User logged in"); // Do NOT log sensitive info like passwords!
    return 0;
}