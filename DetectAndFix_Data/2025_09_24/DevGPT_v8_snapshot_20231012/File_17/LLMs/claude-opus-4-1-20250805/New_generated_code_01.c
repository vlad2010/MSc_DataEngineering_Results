#include <syslog.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_LOG_LENGTH 1024

void secure_log_message(const char *user_input) {
    // Option 1: Use format string properly (RECOMMENDED)
    // Always use a format string with %s for user-controlled data
    syslog(LOG_DEBUG, "%s", user_input);
}

void secure_log_message_with_validation(const char *user_input) {
    // Option 2: With input validation and sanitization
    char sanitized_message[MAX_LOG_LENGTH];
    
    // Validate input
    if (user_input == NULL) {
        syslog(LOG_DEBUG, "%s", "Invalid log message: NULL input");
        return;
    }
    
    // Truncate if too long
    size_t input_len = strlen(user_input);
    if (input_len >= MAX_LOG_LENGTH) {
        input_len = MAX_LOG_LENGTH - 1;
    }
    
    // Copy and sanitize the message
    strncpy(sanitized_message, user_input, input_len);
    sanitized_message[input_len] = '\0';
    
    // Remove or escape potentially dangerous characters
    for (size_t i = 0; i < input_len; i++) {
        // Replace format specifier % with _
        if (sanitized_message[i] == '%') {
            sanitized_message[i] = '_';
        }
        // Ensure printable characters only
        if (sanitized_message[i] < 32 || sanitized_message[i] > 126) {
            sanitized_message[i] = '?';
        }
    }
    
    // Use proper format string
    syslog(LOG_DEBUG, "%s", sanitized_message);
}

void secure_log_with_context(const char *operation, const char *user_input, int user_id) {
    // Option 3: Structured logging with context
    char log_buffer[MAX_LOG_LENGTH];
    
    // Validate inputs
    if (operation == NULL || user_input == NULL) {
        syslog(LOG_ERR, "Invalid parameters for logging");
        return;
    }
    
    // Create structured log message with proper formatting
    int written = snprintf(log_buffer, sizeof(log_buffer), 
                          "Operation: %s | User ID: %d | Message: %s",
                          operation, user_id, user_input);
    
    // Check for truncation
    if (written >= sizeof(log_buffer)) {
        syslog(LOG_WARNING, "Log message truncated");
    }
    
    // Always use format string parameter correctly
    syslog(LOG_DEBUG, "%s", log_buffer);
}

// Example usage
int main() {
    // Initialize syslog
    openlog("secure_app", LOG_PID | LOG_CONS, LOG_USER);
    
    // SECURE: Always use format string
    syslog(LOG_DEBUG, "%s", "Your log message here");
    
    // SECURE: Handle user input safely
    const char *user_message = "User provided message";
    secure_log_message(user_message);
    
    // SECURE: With validation
    secure_log_message_with_validation("Message with validation");
    
    // SECURE: Structured logging
    secure_log_with_context("LOGIN", "User authentication attempt", 12345);
    
    // Close syslog
    closelog();
    
    return 0;
}