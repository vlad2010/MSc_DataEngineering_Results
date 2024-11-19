#include "Adafruit_HUSB238.h"
#include <Adafruit_BusIO.h>
#include <Adafruit_I2CRegister.h>

// Initialize I2C address for HUSB238; replace with actual I2C address
#define HUSB238_I2CADDR_DEFAULT 0x1A  // Example: replace with actual I2C address

Adafruit_HUSB238::Adafruit_HUSB238() {
  // Constructor can be empty if you don't have anything specific here
}

bool Adafruit_HUSB238::begin() {
  // Attempt to initiate I2C communication
  if (!i2c_dev.begin(HUSB238_I2CADDR_DEFAULT)) { 
    // If the device is not available, return false
    return false;
  }

  // Additional setup tasks if necessary
  return true;
}

bool Adafruit_HUSB238::getAttached() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister attachRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);  // Replace with actual register and config
  if (!attachRegister) {
    // Error handling if register creation failed
    return false;
  }
  
  // Create an Adafruit_I2CRegisterBit object for the 7th bit of PD_STATUS0
  Adafruit_I2CRegisterBit attachBit = Adafruit_I2CRegisterBit(&attachRegister, 7);  // 7th bit
  if (!attachBit) {
    // Error handling if bit creation failed
    return false;
  }

  // Read the bit
  bool attachStatus = attachBit.read();
  if (attachStatus == INVALID_BIT_READ) {
    // Error handling if read failed
    return false;
  }

  return attachStatus;
}