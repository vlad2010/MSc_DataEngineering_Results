// Fixed implementations for Adafruit_HUSB238

#include <stdint.h>

// Assume these are declared elsewhere in your project:
// - class Adafruit_I2CDevice;
// - class Adafruit_I2CRegister with ctor Adafruit_I2CRegister(Adafruit_I2CDevice*, uint16_t)
//   and bool read(uint8_t* value);
// - #define or const uint16_t HUSB238_PD_STATUS1

class Adafruit_HUSB238 {
public:
  // Safer API: returns success/failure; writes the bit value to out_value.
  // This avoids conflating "false because read failed" with "false because bit=0".
  bool get5VContractV(bool& out_value);

  // Backward-compatible wrapper: returns the bit value; on error, returns false (fail-closed).
  bool get5VContractV();

private:
  Adafruit_I2CDevice* i2c_dev = nullptr; // must be initialized elsewhere
};

// 0-based bit index for the "second bit" in HUSB238_PD_STATUS1
static constexpr uint8_t HUSB238_PD_STATUS1_5V_CONTRACT_BIT = 1; // bit1 == "2nd bit"
static_assert(HUSB238_PD_STATUS1_5V_CONTRACT_BIT < 8, "Bit index out of range");

// Safer API implementation
bool Adafruit_HUSB238::get5VContractV(bool& out_value) {
  // Validate device handle (prevents CWE-476)
  if (i2c_dev == nullptr) {
    return false;
  }

  // Read the entire status register and check for I2C errors (prevents CWE-252)
  Adafruit_I2CRegister pdRegister(i2c_dev, HUSB238_PD_STATUS1);
  uint8_t status = 0;
  if (!pdRegister.read(&status)) {
    return false; // I2C read failed
  }

  // Extract the intended bit (avoid off-by-one; CWE-193)
  out_value = ((status >> HUSB238_PD_STATUS1_5V_CONTRACT_BIT) & 0x01) != 0;
  return true;
}

// Backward-compatible wrapper that fails closed on error
bool Adafruit_HUSB238::get5VContractV() {
  bool val = false;
  // If the safe read fails, return false (indicates "no 5V contract" as a safe default)
  (void)get5VContractV(val);
  return val;
}