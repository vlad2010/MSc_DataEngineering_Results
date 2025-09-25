#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// Constants for validation
#define MAX_PAYLOAD_LENGTH 10
#define MAX_HOUR 23
#define MAX_MINUTE 59
#define MAX_SECOND 59

// Helper function to safely parse integer from payload
bool safeParseInt(const byte* payload, unsigned int length, int* result, int minVal, int maxVal) {
    // Check for reasonable payload length
    if (length == 0 || length > MAX_PAYLOAD_LENGTH) {
        return false;
    }
    
    // Create null-terminated string from payload
    char buffer[MAX_PAYLOAD_LENGTH + 1];
    memcpy(buffer, payload, length);
    buffer[length] = '\0';
    
    // Validate that all characters are digits (with optional leading sign)
    int start = 0;
    if (buffer[0] == '-' || buffer[0] == '+') {
        if (length == 1) return false; // Just a sign, no digits
        start = 1;
    }
    
    for (unsigned int i = start; i < length; i++) {
        if (!isdigit(buffer[i])) {
            return false;
        }
    }
    
    // Parse the integer using strtol for better error handling
    char* endptr;
    errno = 0;
    long value = strtol(buffer, &endptr, 10);
    
    // Check for conversion errors
    if (errno == ERANGE || *endptr != '\0') {
        return false;
    }
    
    // Check if value is within int range and specified bounds
    if (value < minVal || value > maxVal || value > INT_MAX || value < INT_MIN) {
        return false;
    }
    
    *result = (int)value;
    return true;
}

void callback(char* topic, byte* payload, unsigned int length) {
    // Validate topic pointer
    if (topic == NULL || payload == NULL) {
        return;
    }
    
    // Temporary variable for parsed value
    int parsedValue = 0;
    
    // Check which topic was received and parse accordingly
    if (strcmp(topic, "@msg/timer/setHr") == 0) {
        if (safeParseInt(payload, length, &parsedValue, 0, MAX_HOUR)) {
            receivedHour = parsedValue;
            // Optional: Add logging for successful update
            // Serial.printf("Hour updated to: %d\n", receivedHour);
        } else {
            // Optional: Add error logging
            // Serial.println("Invalid hour value received");
        }
    } else if (strcmp(topic, "@msg/timer/setMinute") == 0) {
        if (safeParseInt(payload, length, &parsedValue, 0, MAX_MINUTE)) {
            receivedMinute = parsedValue;
            // Optional: Add logging for successful update
            // Serial.printf("Minute updated to: %d\n", receivedMinute);
        } else {
            // Optional: Add error logging
            // Serial.println("Invalid minute value received");
        }
    } else if (strcmp(topic, "@msg/timer/setSec") == 0) {
        if (safeParseInt(payload, length, &parsedValue, 0, MAX_SECOND)) {
            receivedSecond = parsedValue;
            // Optional: Add logging for successful update
            // Serial.printf("Second updated to: %d\n", receivedSecond);
        } else {
            // Optional: Add error logging
            // Serial.println("Invalid second value received");
        }
    }
    // Silently ignore unknown topics (or add logging if needed)
}