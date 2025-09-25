#include "Adafruit_HUSB238.h"
#include <Adafruit_BusIO.h>
#include <Adafruit_I2CRegister.h>

// Remove hardcoded I2C address
// #define HUSB238_I2CADDR_DEFAULT 0xXX

class Adafruit_HUSB238 {
public:
  Adafruit_HUSB238(uint8_t i2c_addr);
  bool begin(TwoWire *theWire = &Wire);
  bool getAttached();

private:
  uint8_t _i2c_addr;
  Adafruit_I2CDevice *i2c_dev = nullptr;
};

Adafruit_HUSB238::Adafruit_HUSB238(uint8_t i2c_addr) : _i2c_addr(i2c_addr) {
  // Constructor
}

bool Adafruit_HUSB238::begin(TwoWire *theWire) {
  // Initialize I2C device
  if (i2c_dev) {
    delete i2c_dev;
    i2c_dev = nullptr;
  }
  i2c_dev = new Adafruit_I2CDevice(_i2c_addr, theWire);
  if (!i2c_dev->begin()) {
    // Could not initialize I2C device
    return false;
  }
  return true;
}

bool Adafruit_HUSB238::getAttached() {
  if (!i2c_dev || !i2c_dev->begin()) {
    // Device not initialized
    return false;
  }

  // Replace HUSB238_PD_STATUS0 with actual register address
  Adafruit_I2CRegister attachRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);

  // Check if register is valid (if API supports it)
  // if (!attachRegister.isValid()) return false;

  Adafruit_I2CRegisterBit attachBit = Adafruit_I2CRegisterBit(&attachRegister, 7);

  bool attachStatus = false;
  if (!attachBit.read(&attachStatus)) {
    // Read failed
    return false;
  }

  return attachStatus;
}