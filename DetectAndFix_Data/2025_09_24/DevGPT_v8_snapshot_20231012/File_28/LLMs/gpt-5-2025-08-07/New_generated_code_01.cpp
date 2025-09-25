// Assumed environment: Adafruit BusIO, Arduino C++
// Adjust enum values to match the actual HUSB238 datasheet / library definitions.

#include <Adafruit_I2CDevice.h>
#include <Adafruit_BusIO_Register.h>
#include <stdint.h>

// Device register address for PD status
#ifndef HUSB238_PD_STATUS1
#define HUSB238_PD_STATUS1 0x01  // Replace with the correct address if different
#endif

// Strongly-typed enum for response codes; include a safe fallback Unknown
enum class HUSB238_ResponseCodes : uint8_t {
  GoodCRC = 0,   // Replace with actual code values as per the datasheet
  Accept  = 1,
  Reject  = 2,
  Ping    = 3,
  PS_RDY  = 4,
  // ... add other valid codes here ...
  Unknown = 0xFF
};

class Adafruit_HUSB238 {
public:
  // Existing member provided by the library; ensure it's initialized by begin()
  Adafruit_I2CDevice* i2c_dev = nullptr;

  // Safer API: returns success, writes result to out parameter
  bool getPDResponse(HUSB238_ResponseCodes& out);

  // Backward-compatible wrapper: returns Unknown on failure
  HUSB238_ResponseCodes getPDResponseLegacy();

private:
  static bool mapRawToResponse(uint8_t raw, HUSB238_ResponseCodes& out);
};

// Map raw 3-bit field to a defined response code; reject unknown values.
// Update cases to match the actual device's encoding.
bool Adafruit_HUSB238::mapRawToResponse(uint8_t raw, HUSB238_ResponseCodes& out) {
  switch (raw) {
    case 0: out = HUSB238_ResponseCodes::GoodCRC; return true;
    case 1: out = HUSB238_ResponseCodes::Accept;  return true;
    case 2: out = HUSB238_ResponseCodes::Reject;  return true;
    case 3: out = HUSB238_ResponseCodes::Ping;    return true;
    case 4: out = HUSB238_ResponseCodes::PS_RDY;  return true;
    // Add other known valid raw values here...
    default:
      out = HUSB238_ResponseCodes::Unknown;
      return false;
  }
}

bool Adafruit_HUSB238::getPDResponse(HUSB238_ResponseCodes& out) {
  // CWE-476: guard against null device
  if (i2c_dev == nullptr) {
    out = HUSB238_ResponseCodes::Unknown;
    return false;
  }

  // Read the full register with error checking (avoid Bits helper that hides status)
  Adafruit_I2CRegister pdRegister(i2c_dev, HUSB238_PD_STATUS1, 1 /* width bytes */);

  uint32_t regVal = 0;
  if (!pdRegister.read(&regVal)) {  // CWE-252: check I2C read result
    out = HUSB238_ResponseCodes::Unknown;
    return false;
  }

  // Extract bits 3..5 safely
  const uint8_t raw = static_cast<uint8_t>((regVal >> 3) & 0x07);

  // Validate and convert to enum (CWE-704 mitigation)
  if (!mapRawToResponse(raw, out)) {
    // You can decide whether an unknown code is a hard failure:
    // return false; // if you want to signal error for unknown values
    return true;      // or treat as success with Unknown sentinel
  }

  return true;
}

// Backward-compatible wrapper that returns a safe sentinel on failure.
HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponseLegacy() {
  HUSB238_ResponseCodes rc = HUSB238_ResponseCodes::Unknown;
  if (!getPDResponse(rc)) {
    // Optionally log/trace here
    return HUSB238_ResponseCodes::Unknown;
  }
  return rc;
}