#include "Adafruit_HUSB238.h"

Adafruit_HUSB238::Adafruit_HUSB238() : i2c_dev(nullptr) {}

Adafruit_HUSB238::~Adafruit_HUSB238() {
  if (i2c_dev) {
    delete i2c_dev;
    i2c_dev = nullptr;  // Avoid dangling pointer
  }
}

bool Adafruit_HUSB238::begin(uint8_t addr, TwoWire *theWire) {
  if (i2c_dev) {
    delete i2c_dev;
  }

  i2c_dev = new (std::nothrow) Adafruit_I2CDevice(addr, theWire);  // Use std::nothrow for failure check
  if (i2c_dev == nullptr) {
    return false;  // Allocation failed
  }

  if (!i2c_dev->begin()) {
    delete i2c_dev;  // Clean up if initialization fails
    i2c_dev = nullptr;
    return false;
  }

  return true;
}

bool Adafruit_HUSB238::getCCStatus() {
  Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 7);
  return ccBit.read();
}

bool Adafruit_HUSB238::getAttached() {
  Adafruit_I2CRegister attachRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit attachBit = Adafruit_I2CRegisterBit(&attachRegister, 6);
  return attachBit.read();
}