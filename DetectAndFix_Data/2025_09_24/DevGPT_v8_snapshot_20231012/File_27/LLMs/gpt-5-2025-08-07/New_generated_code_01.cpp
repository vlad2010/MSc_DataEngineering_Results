#include <Adafruit_I2CDevice.h>
#include <Adafruit_BusIO_Register.h>

// Assuming HUSB238_PD_STATUS1 is defined elsewhere in your project.
#ifndef HUSB238_PD_STATUS1
#define HUSB238_PD_STATUS1 0x00  // Replace with the correct address if not already defined
#endif

class Adafruit_HUSB238 {
public:
  explicit Adafruit_HUSB238(Adafruit_I2CDevice* dev) : i2c_dev(dev) {}

  // Safer API: returns true on success and sets 'cc'; returns false on error (I2C/read failure/null device).
  bool tryGetCCStatus(bool& cc) {
    last_error_ = Error::None;

    if (i2c_dev == nullptr) {
      last_error_ = Error::NullDevice;
      return false; // CWE-476 mitigation
    }

    // Read the entire register so we can detect I2C errors (CWE-252/CWE-703 mitigation)
    Adafruit_I2CRegister reg(i2c_dev, HUSB238_PD_STATUS1 /*, width = 1 byte default */);
    uint8_t regval = 0;
    if (!reg.read(&regval)) {
      last_error_ = Error::I2CReadFailure;
      return false;
    }

    static constexpr uint8_t HUSB238_PD_STATUS1_CC_BIT = 6; // Named constant instead of magic number
    static_assert(HUSB238_PD_STATUS1_CC_BIT < 8, "CC bit must be within an 8-bit register");
    cc = ((regval >> HUSB238_PD_STATUS1_CC_BIT) & 0x01) != 0;
    return true;
  }

  // Backward compatible API: returns the CC bit; fails closed (false) if read fails.
  // Note: This conflates "false" with "error". Prefer tryGetCCStatus above when possible.
  bool getCCStatus() {
    bool cc = false;
    if (!tryGetCCStatus(cc)) {
      // Fail-closed behavior: treat as 'not connected' on error
      return false;
    }
    return cc;
  }

  enum class Error {
    None,
    NullDevice,
    I2CReadFailure
  };

  Error lastError() const { return last_error_; }

private:
  Adafruit_I2CDevice* i2c_dev = nullptr;
  Error last_error_ = Error::None;
};