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