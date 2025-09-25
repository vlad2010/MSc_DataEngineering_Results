#include <stdint.h>

// Assuming Adafruit BusIO is available
#include <Adafruit_I2CDevice.h>

// Device-specific register address (should be defined elsewhere in your project)
#ifndef HUSB238_PD_STATUS1
#define HUSB238_PD_STATUS1 0x33  // Replace with the correct address if different
#endif

class Adafruit_HUSB238 {
public:
  enum class Error {
    None = 0,
    NotInitialized,
    I2CError
  };

  explicit Adafruit_HUSB238(Adafruit_I2CDevice* dev) : i2c_dev(dev) {}

  // Returns true if attached bit is set, false otherwise.
  // On failure, returns false and sets last_error_ to a non-None value.
  bool getAttached();

  // Allow callers to inspect the last error to disambiguate false vs failure.
  Error lastError() const { return last_error_; }

private:
  static constexpr uint8_t kAttachBitIndex = 6; // 6th bit
  static_assert(kAttachBitIndex < 8, "Attach bit index must be within [0,7] for 8-bit register");

  // Helper to read one 8-bit register with basic retry
  bool readReg8(uint8_t reg, uint8_t& out) {
    if (!i2c_dev) return false;

    constexpr uint8_t max_attempts = 3;
    uint8_t attempts = 0;
    while (attempts++ < max_attempts) {
      uint8_t cmd = reg;
      // Adafruit_I2CDevice::write_then_read returns bool (true on success)
      if (i2c_dev->write_then_read(&cmd, 1, &out, 1)) {
        return true;
      }
      // Small delay could be added here if available (e.g., delay(1))
    }
    return false;
  }

  Adafruit_I2CDevice* i2c_dev = nullptr;
  Error last_error_ = Error::None;
};

bool Adafruit_HUSB238::getAttached() {
  last_error_ = Error::None;

  // CWE-476: guard against null device
  if (!i2c_dev) {
    last_error_ = Error::NotInitialized;
    return false;
  }

  // Read the status register with proper error handling
  uint8_t status = 0;
  if (!readReg8(HUSB238_PD_STATUS1, status)) {
    // CWE-252 / CWE-703: handle I2C read failure
    last_error_ = Error::I2CError;
    return false; // Caller can inspect lastError() to differentiate failure vs "not attached"
  }

  // Safely extract the bit
  bool attached = ((status >> kAttachBitIndex) & 0x01u) != 0;
  return attached;
}