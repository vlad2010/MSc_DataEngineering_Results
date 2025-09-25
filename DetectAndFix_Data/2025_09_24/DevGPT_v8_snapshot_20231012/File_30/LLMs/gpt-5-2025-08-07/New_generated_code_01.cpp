#include <cstdint>

// Forward declarations of Adafruit classes (include the real headers in your project)
#include <Adafruit_I2CDevice.h>
#include <Adafruit_I2CRegister.h>
#include <Adafruit_I2CRegisterBits.h>

// Replace with the actual register address from your hardware definition
#ifndef HUSB238_PD_STATUS1
#define HUSB238_PD_STATUS1 0x00
#endif

// If your enum is already defined elsewhere, remove or adjust this.
// Using an enum class with an explicit underlying type improves type safety.
enum class HUSB238_5VCurrentContract : uint8_t {
  // Replace names/values with your real enum enumerators if they differ.
  Contract0 = 0, // e.g., 500 mA
  Contract1 = 1, // e.g., 900 mA
  Contract2 = 2, // e.g., 1.5 A
  Contract3 = 3, // e.g., 3.0 A
  // Optional: add an explicit Unknown to represent failures/out-of-range safely.
  Unknown   = 0xFF
};

class Adafruit_HUSB238 {
public:
  HUSB238_5VCurrentContract get5VContractA();

private:
  Adafruit_I2CDevice* i2c_dev = nullptr; // must be initialized elsewhere
};

// Helper to map 2-bit value to enum safely.
// If your enum does not have an Unknown value, you can change the default to a safe choice.
static inline HUSB238_5VCurrentContract contractFromBits(uint8_t bits) noexcept {
  switch (bits & 0x03u) {
    case 0: return static_cast<HUSB238_5VCurrentContract>(0);
    case 1: return static_cast<HUSB238_5VCurrentContract>(1);
    case 2: return static_cast<HUSB238_5VCurrentContract>(2);
    case 3: return static_cast<HUSB238_5VCurrentContract>(3);
    default: return HUSB238_5VCurrentContract::Unknown;
  }
}

HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  // CWE-476: guard against null I2C device pointer
  if (i2c_dev == nullptr) {
    return HUSB238_5VCurrentContract::Unknown; // fail safely
  }

  Adafruit_I2CRegister pdRegister(i2c_dev, HUSB238_PD_STATUS1);

  // Prefer explicit error handling if available in your library version.
  // Some versions of Adafruit_I2CRegister expose bool read(uint8_t*, size_t) or bool read(uint32_t*).
  // If available, uncomment the following and remove the fallback:
  //
  // uint32_t regval = 0;
  // if (!pdRegister.read(&regval)) { // check for I2C error
  //   return HUSB238_5VCurrentContract::Unknown; // CWE-252: handle read failure
  // }
  //
  // Fallback: read and assume 0 on failure (documented limitation).
  uint32_t regval = pdRegister.read();

  // Extract the bottom two bits (0-1) safely.
  uint8_t bits = static_cast<uint8_t>(regval & 0x03u);

  // CWE-704/CWE-20: validate and map to enum instead of a raw C-style cast.
  return contractFromBits(bits);
}