#include <cstring>
#include <cstdlib>
#include <cctype>
#include <algorithm>

#define MAX_PAYLOAD_SIZE 16 // Adjust as needed

int receivedHour = 0;
int receivedMinute = 0;
int receivedSecond = 0;

// Helper function to safely parse integer from payload
int safeParseInt(const byte* payload, unsigned int length, int minVal, int maxVal) {
    char buffer[MAX_PAYLOAD_SIZE];
    unsigned int copyLen = std::min(length, (unsigned int)(MAX_PAYLOAD_SIZE - 1));
    memcpy(buffer, payload, copyLen);
    buffer[copyLen] = '\0';

    // Validate that buffer contains only digits (and optional leading +/-)
    char* ptr = buffer;
    if (*ptr == '+' || *ptr == '-') ++ptr;
    for (; *ptr; ++ptr) {
        if (!isdigit(*ptr)) return minVal - 1; // Invalid input
    }

    int value = atoi(buffer);
    if (value < minVal || value > maxVal) return minVal - 1; // Out of range
    return value;
}

void callback(char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, "@msg/timer/setHr") == 0) {
        int val = safeParseInt(payload, length, 0, 23);
        if (val >= 0) receivedHour = val;
        // else: handle error
    } else if (strcmp(topic, "@msg/timer/setMinute") == 0) {
        int val = safeParseInt(payload, length, 0, 59);
        if (val >= 0) receivedMinute = val;
        // else: handle error
    } else if (strcmp(topic, "@msg/timer/setSec") == 0) {
        int val = safeParseInt(payload, length, 0, 59);
        if (val >= 0) receivedSecond = val;
        // else: handle error
    }
}