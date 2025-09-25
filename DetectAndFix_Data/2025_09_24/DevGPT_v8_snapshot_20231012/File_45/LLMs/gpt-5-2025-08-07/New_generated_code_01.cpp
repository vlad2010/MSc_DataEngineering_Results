#include <stdint.h>
// Include Adafruit BusIO headers as needed
// #include <Adafruit_I2CDevice.h>
// #include <Adafruit_I2CRegister.h>

// Forward declarations for types and register constants used by this class.
// These should already exist in your project; shown here for completeness.
enum HUSB238_ResponseCodes : uint8_t {
  // Fill with actual known values from your device datasheet
  // Example placeholders:
  HUSB238_PD_RESP_0 = 0,
  HUSB238_PD_RESP_1 = 1,
  HUSB238_PD_RESP_2 = 2,
  HUSB238_PD_RESP_3 = 3,
  HUSB238_PD_RESP_4 = 4,
  HUSB238_PD_RESP_5 = 5,
  // Add more as appropriate; we will treat values outside the known range as invalid.
};

enum HUSB238_5VCurrentContract : uint8_t {
  // Example placeholders (2-bit field => 0..3)
  HUSB238_5V_CURR_A0 = 0,
  HUSB238_5V_CURR_A1 = 1,
  HUSB238_5V_CURR_A2 = 2,
  HUSB238_5V_CURR_A3 = 3,
};

#ifndef HUSB238_PD_STATUS1
#define HUSB238_PD_STATUS1 0x00 // Replace with the actual register address
#endif

class Adafruit_I2CDevice;
class Adafruit_I2CRegister;

// Minimal class shell so this compiles as a single fragment.
class Adafruit_HUSB238 {
public:
  // Safe APIs that surface I2C read errors and validate values.
  [[nodiscard]] bool tryGetPDResponse(HUSB238_ResponseCodes &out) noexcept;
  [[nodiscard]] bool tryGet5VContractV(bool &out) noexcept;
  [[nodiscard]] bool tryGet5VContractA(HUSB238_5VCurrentContract &out) noexcept;

  // Backward-compatible wrappers that return conservative defaults on failure.
  HUSB238_ResponseCodes getPDResponse();
  bool get5VContractV();
  HUSB238_5VCurrentContract get5VContractA();

  // ctor elsewhere – ensure i2c_dev is valid before calling any method.
  Adafruit_I2CDevice *i2c_dev = nullptr;

private:
  // Bit field definitions for HUSB238_PD_STATUS1
  static constexpr uint8_t PD_RESP_SHIFT = 3; // bits 3..5
  static constexpr uint8_t PD_RESP_WIDTH = 3; // 3 bits
  static constexpr uint8_t CONTRACT_V_SHIFT = 2; // bit 2
  static constexpr uint8_t CONTRACT_V_WIDTH = 1; // 1 bit
  static constexpr uint8_t CONTRACT_A_SHIFT = 0; // bits 0..1
  static constexpr uint8_t CONTRACT_A_WIDTH = 2; // 2 bits

  // Conservative maximums by bit width. If you know the exact valid set,
  // further restrict these checks.
  static constexpr uint8_t PD_RESP_MAX = (1u << PD_RESP_WIDTH) - 1u; // 0..7
  static constexpr uint8_t CONTRACT_A_MAX = (1u << CONTRACT_A_WIDTH) - 1u; // 0..3

  [[nodiscard]] bool readReg8(uint8_t reg, uint8_t &out) noexcept;
  [[nodiscard]] bool readBits(uint8_t reg, uint8_t shift, uint8_t width, uint8_t &out) noexcept;

  template <typename E>
  [[nodiscard]] static bool validateEnumRange(uint8_t raw, uint8_t maxAllowed) noexcept {
    // If you can enumerate the exact known-valid values, do that instead of max range.
    return raw <= maxAllowed;
  }
};

#include <Adafruit_I2CRegister.h> // Now we can use it

bool Adafruit_HUSB238::readReg8(uint8_t reg, uint8_t &out) noexcept {
  // Use Adafruit_I2CRegister to get an error-aware read.
  // This avoids the helper `Bits` type so we can check the return status.
  Adafruit_I2CRegister pdRegister(i2c_dev, reg);
  return pdRegister.read(&out); // returns bool success
}

bool Adafruit_HUSB238::readBits(uint8_t reg, uint8_t shift, uint8_t width, uint8_t &out) noexcept {
  uint8_t val = 0;
  if (!readReg8(reg, val)) {
    return false;
  }
  const uint8_t mask = static_cast<uint8_t>(((1u << width) - 1u) << shift);
  out = static_cast<uint8_t>((val & mask) >> shift);
  return true;
}

bool Adafruit_HUSB238::tryGetPDResponse(HUSB238_ResponseCodes &out) noexcept {
  uint8_t raw = 0;
  if (!readBits(HUSB238_PD_STATUS1, PD_RESP_SHIFT, PD_RESP_WIDTH, raw)) {
    return false; // I2C read failed
  }
  if (!validateEnumRange<HUSB238_ResponseCodes>(raw, PD_RESP_MAX)) {
    return false; // Out-of-range or reserved value encountered
  }
  out = static_cast<HUSB238_ResponseCodes>(raw); // Safe cast after validation
  return true;
}

bool Adafruit_HUSB238::tryGet5VContractV(bool &out) noexcept {
  uint8_t raw = 0;
  if (!readBits(HUSB238_PD_STATUS1, CONTRACT_V_SHIFT, CONTRACT_V_WIDTH, raw)) {
    return false;
  }
  // raw is 0 or 1 due to width 1
  out = (raw != 0);
  return true;
}

bool Adafruit_HUSB238::tryGet5VContractA(HUSB238_5VCurrentContract &out) noexcept {
  uint8_t raw = 0;
  if (!readBits(HUSB238_PD_STATUS1, CONTRACT_A_SHIFT, CONTRACT_A_WIDTH, raw)) {
    return false;
  }
  if (!validateEnumRange<HUSB238_5VCurrentContract>(raw, CONTRACT_A_MAX)) {
    return false;
  }
  out = static_cast<HUSB238_5VCurrentContract>(raw);
  return true;
}

// Backward-compatible wrappers. They return conservative defaults on failure.
// Consider deprecating these in favor of the tryGet* versions.

HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponse() {
  HUSB238_ResponseCodes value{};
  if (tryGetPDResponse(value)) {
    return value;
  }
  // Conservative fallback: return 0 (commonly a benign/reset-like state).
  // If you have a specific "Unknown" or "Invalid" enumerator, return that instead.
  return static_cast<HUSB238_ResponseCodes>(0);
}

bool Adafruit_HUSB238::get5VContractV() {
  bool value = false;
  if (tryGet5VContractV(value)) {
    return value;
  }
  // On failure, return false (safer default).
  return false;
}

HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  HUSB238_5VCurrentContract value{};
  if (tryGet5VContractA(value)) {
    return value;
  }
  // Conservative fallback: return the lowest current level.
  return static_cast<HUSB238_5VCurrentContract>(0);
}