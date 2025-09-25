// Single-file example that fixes the security issues found.
// Assumes Arduino environment with Adafruit BusIO available.

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BusIO.h>
#include <Adafruit_I2CRegister.h>

// Replace these with the correct values for your HUSB238.
// Default 7-bit I2C address (example value; consult datasheet).
constexpr uint8_t HUSB238_I2CADDR_DEFAULT = 0x08;  // TODO: set actual address
// PD_STATUS0 register address (example value; consult datasheet).
constexpr uint8_t HUSB238_PD_STATUS0      = 0x00;  // TODO: set actual register

class Adafruit_HUSB238 {
public:
  Adafruit_HUSB238();
  ~Adafruit_HUSB238();

  // Safe begin: validates address, initializes I2C device, and probes it.
  bool begin(uint8_t i2c_addr = HUSB238_I2CADDR_DEFAULT, TwoWire *wire = &Wire);

  // Preferred API: returns true on success and sets 'attached'.
  // If it returns false, 'attached' is not modified.
  bool getAttached(bool &attached);

  // Legacy API with original signature:
  // Returns "attached" state, but returns false also on error (ambiguous).
  // Prefer the overload above for reliable error handling.
  bool getAttachedLegacy();

  bool isInitialized() const { return _initialized; }

private:
  Adafruit_I2CDevice *_i2c_dev = nullptr;
  bool _initialized = false;
  uint8_t _addr = 0;
  TwoWire *_wire = nullptr;
};

Adafruit_HUSB238::Adafruit_HUSB238() {}

Adafruit_HUSB238::~Adafruit_HUSB238() {
  if (_i2c_dev) {
    delete _i2c_dev;
    _i2c_dev = nullptr;
  }
  _initialized = false;
}

bool Adafruit_HUSB238::begin(uint8_t i2c_addr, TwoWire *wire) {
  // Validate parameters
  if (wire == nullptr) {
    return false; // CWE-476 mitigation: don't dereference null wire
  }
  // Validate 7-bit I2C address range (0x08..0x77 commonly valid)
  if (i2c_addr < 0x08 || i2c_addr > 0x77) {
    return false; // CWE-665: reject improper initialization parameters
  }

  if (_initialized) {
    // Already initialized; no-op success
    return true;
  }

  _wire = wire;
  _addr = i2c_addr;

  // Ensure Wire is started. Some cores require this; harmless if already called.
  _wire->begin();
#if defined(WIRE_HAS_TIMEOUT)
  _wire->setTimeout(50); // Avoid long bus hangs; tune as needed
#endif

  // Allocate I2C device (nothrow to avoid exceptions on embedded)
  _i2c_dev = new (std::nothrow) Adafruit_I2CDevice(_addr, _wire);
  if (_i2c_dev == nullptr) {
    return false; // CWE-401: avoid continuing on allocation failure
  }

  // Probe the device
  if (!_i2c_dev->begin()) {
    delete _i2c_dev;
    _i2c_dev = nullptr;
    return false; // CWE-393/703: correct status reporting
  }

  _initialized = true;
  return true;
}

bool Adafruit_HUSB238::getAttached(bool &attached) {
  // CWE-476/CWE-665: ensure initialized before use
  if (!_initialized || _i2c_dev == nullptr) {
    return false;
  }

  // Read the PD_STATUS0 register byte safely and check for I2C errors
  Adafruit_I2CRegister statusReg(_i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);
  uint8_t status = 0;
  bool ok = statusReg.read(&status);
  if (!ok) {
    return false; // CWE-252: propagate read failure to caller
  }

  // 7th bit (bit index 7) is the "attached" bit per original code
  attached = ((status >> 7) & 0x01) != 0;
  return true;
}

bool Adafruit_HUSB238::getAttachedLegacy() {
  bool attached = false;
  if (!getAttached(attached)) {
    // Ambiguity: return false on error OR "not attached".
    // Keep this behavior only for backward compatibility.
    return false;
  }
  return attached;
}

// Example usage:
//
// Adafruit_HUSB238 husb;
// void setup() {
//   Serial.begin(115200);
//   if (!husb.begin(HUSB238_I2CADDR_DEFAULT, &Wire)) {
//     Serial.println("HUSB238 init failed!");
//     while (1) delay(1000);
//   }
//   bool attached;
//   if (husb.getAttached(attached)) {
//     Serial.print("Attached: "); Serial.println(attached ? "yes" : "no");
//   } else {
//     Serial.println("Failed to read attach status");
//   }
// }
//
// void loop() {}