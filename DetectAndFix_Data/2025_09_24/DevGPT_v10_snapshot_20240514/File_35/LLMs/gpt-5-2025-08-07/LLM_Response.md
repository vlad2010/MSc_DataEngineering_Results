Here’s a security-focused review and fixes from a senior software/security perspective.

Key issues identified:
- CWE-170: Improper Null Termination. The payload from MQTT is not guaranteed to be null-terminated. Casting byte* to char* and passing to atoi expects a null-terminated C-string. This can make atoi read past the provided buffer.
- CWE-125: Out-of-bounds Read. Because of the missing terminator, atoi may read beyond the payload length into adjacent memory, causing undefined behavior.
- CWE-20: Improper Input Validation. The code accepts any input and does not validate that it’s numeric or within a valid range (e.g., hour 0–23, minute/second 0–59).
- CWE-704: Incorrect Type Conversion or Cast. Casting a raw byte buffer to char* and treating it as a string without ensuring constraints leads to misuse of APIs that expect strings.
- CWE-190: Integer Overflow or Wraparound (theoretical). Extremely long numeric payloads could overflow intermediate computations without validation. Practical risk is low here but best handled by validating length and range.
- CWE-252: Unchecked Return Value. atoi provides no error reporting; 0 can mean “0” or “failed to parse”.

How to fix:
- Do not treat payload as a C-string; use the provided length.
- Validate that the payload is strictly numeric and within a small maximum length.
- Parse numerically without requiring a null terminator.
- Enforce strict ranges for hour/minute/second before assignment.
- Provide clear error handling/logging and ignore invalid inputs.

Fixed code (single fragment):
```cpp
#include <Arduino.h>

// Safer types for time fields
static uint8_t receivedHour = 0;
static uint8_t receivedMinute = 0;
static uint8_t receivedSecond = 0;

// Parse an unsigned integer from a byte buffer without requiring null-termination.
// Returns true on success and writes the parsed value to outVal.
// Enforces that all characters are decimal digits and that length is reasonable.
static bool parseUnsigned(const byte* payload, unsigned int length, unsigned int& outVal) {
  // Reject empty or excessively long numeric payloads
  // For time values, 1–3 digits is more than enough; we accept up to 3 to be defensive.
  if (payload == nullptr || length == 0 || length > 3) {
    return false;
  }

  unsigned int value = 0;
  for (unsigned int i = 0; i < length; ++i) {
    char c = static_cast<char>(payload[i]);
    if (c < '0' || c > '9') {
      return false; // Non-digit character found
    }
    unsigned int digit = static_cast<unsigned int>(c - '0');

    // Prevent overflow on accumulation (defensive; value is tiny in practice)
    if (value > (UINT_MAX - digit) / 10) {
      return false;
    }

    value = value * 10 + digit;
  }

  outVal = value;
  return true;
}

// Validate parsed value against an inclusive range
static bool validateRange(unsigned int value, unsigned int minV, unsigned int maxV) {
  return (value >= minV && value <= maxV);
}

void callback(char* topic, byte* payload, unsigned int length) {
  // ข้อความ callback สำหรับตั้งค่าเวลา: ชั่วโมง/นาที/วินาที
  // Security fixes:
  // - Do not cast payload to char* and pass to atoi (no null-termination).
  // - Validate content is numeric and within allowed ranges before assignment.

  if (topic == nullptr) {
    return; // Defensive: invalid topic pointer
  }

  unsigned int parsed = 0;

  if (strcmp(topic, "@msg/timer/setHr") == 0) {
    if (parseUnsigned(payload, length, parsed) && validateRange(parsed, 0, 23)) {
      receivedHour = static_cast<uint8_t>(parsed);
    } else {
      // Optional: log or handle invalid input
      // Serial.println(F("Invalid hour payload"));
    }
  } else if (strcmp(topic, "@msg/timer/setMinute") == 0) {
    if (parseUnsigned(payload, length, parsed) && validateRange(parsed, 0, 59)) {
      receivedMinute = static_cast<uint8_t>(parsed);
    } else {
      // Serial.println(F("Invalid minute payload"));
    }
  } else if (strcmp(topic, "@msg/timer/setSec") == 0) {
    if (parseUnsigned(payload, length, parsed) && validateRange(parsed, 0, 59)) {
      receivedSecond = static_cast<uint8_t>(parsed);
    } else {
      // Serial.println(F("Invalid second payload"));
    }
  }
}
```

Notes:
- Eliminated atoi and any reliance on null-terminated strings to avoid CWE-170/CWE-125.
- Strict numeric and range validation mitigates CWE-20 and reduces risk of CWE-190.
- Kept strcmp for topic because PubSubClient typically supplies null-terminated topics; if uncertain, you could add a length-aware comparison from the MQTT library or normalize topics earlier.
- Consider authentication/authorization at the MQTT layer to prevent untrusted control of device time (architectural control, not shown here).