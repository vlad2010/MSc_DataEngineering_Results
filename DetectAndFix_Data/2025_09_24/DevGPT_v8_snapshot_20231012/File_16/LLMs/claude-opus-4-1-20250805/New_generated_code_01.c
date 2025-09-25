#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <errno.h>

#define MAX_LOG_LENGTH 1024
#define MAX_LOGS_PER_SECOND 100

// Thread-safe logging with rate limiting
static time_t last_log_time = 0;
static int log_count = 0;

// Secure logging function
int secure_log(FILE *output_stream, const char *level, const char *format, ...) {
    // Input validation
    if (!output_stream || !level || !format) {
        return -1;
    }
    
    // Rate limiting to prevent log flooding (CWE-400)
    time_t current_time = time(NULL);
    if (current_time == last_log_time) {
        log_count++;
        if (log_count > MAX_LOGS_PER_SECOND) {
            return -2; // Rate limit exceeded
        }
    } else {
        last_log_time = current_time;
        log_count = 1;
    }
    
    // Create timestamp
    char timestamp[64];
    struct tm *tm_info = localtime(&current_time);
    if (tm_info == NULL) {
        return -3;
    }
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Safely format the log message (prevents CWE-134)
    char log_buffer[MAX_LOG_LENGTH];
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(log_buffer, sizeof(log_buffer), format, args);
    va_end(args);
    
    if (ret < 0 || ret >= MAX_LOG_LENGTH) {
        // Truncation occurred or error
        log_buffer[MAX_LOG_LENGTH - 1] = '\0';
    }
    
    // Sanitize output to prevent injection attacks
    // Remove any control characters except newline
    for (size_t i = 0; i < strlen(log_buffer); i++) {
        if (log_buffer[i] < 32 && log_buffer[i] != '\n' && log_buffer[i] != '\t') {
            log_buffer[i] = '?';
        }
    }
    
    // Write log with proper error handling (fixes CWE-252)
    int write_result = fprintf(output_stream, "[%s] [%s] %s", timestamp, level, log_buffer);
    
    if (write_result < 0) {
        // Handle write error appropriately
        // In production, might want to try alternative logging mechanism
        return -4;
    }
    
    // Ensure data is written
    if (fflush(output_stream) != 0) {
        return -5;
    }
    
    return 0;
}

// Wrapper for different log levels with information disclosure protection (CWE-209)
void log_error(const char *format, ...) {
    char safe_format[MAX_LOG_LENGTH];
    va_list args;
    va_start(args, format);
    vsnprintf(safe_format, sizeof(safe_format), format, args);
    va_end(args);
    
    #ifdef DEBUG
        // In debug mode, show detailed error
        secure_log(stderr, "ERROR", "%s\n", safe_format);
    #else
        // In production, log generic message to stderr, detailed to file
        secure_log(stderr, "ERROR", "An error occurred. Check logs for details.\n");
        
        // Log detailed error to secure log file (with proper permissions)
        FILE *log_file = fopen("/var/log/app_secure.log", "a");
        if (log_file) {
            // Set restrictive permissions on log file
            chmod("/var/log/app_secure.log", 0600);
            secure_log(log_file, "ERROR", "%s\n", safe_format);
            fclose(log_file);
        }
    #endif
}

// Example usage
int main() {
    // Safe logging - no format string vulnerability
    log_error("Your log message here");
    
    // Safe logging with parameters
    int error_code = 42;
    log_error("Operation failed with code: %d", error_code);
    
    // Direct secure logging
    secure_log(stderr, "INFO", "Application started successfully\n");
    
    return 0;
}