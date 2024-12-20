/**************************************************************************/
/*!
  @file     Adafruit_HUSB238.cpp

  @mainpage Adafruit HUSB238 I2C USB PD sink

  @section intro Introduction

  This is a library for the Adafruit HUSB238 I2C USB PD sink
  ----> http://www.adafruit.com/

  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  @section author Author

  Limor Fied (Adafruit Industries)

  @section license License

  BSD (see license.txt)
*/
/**************************************************************************/
#include "Adafruit_HUSB238.h"

/**************************************************************************/
/*!
    @brief  Instantiates a new HUSB238 class
*/
/**************************************************************************/
Adafruit_HUSB238::Adafruit_HUSB238() {}

/**************************************************************************/
/*!
    @brief  Sets up the I2C connection and tests that the sensor was found.
    @param addr The 7-bit I2C address of the HUSB238
    @param theWire Pointer to an I2C device we'll use to communicate
    default is Wire
    @return true if sensor was found, otherwise false.
    @details This function initializes the I2C communication with the HUSB238 device.
    It uses the provided I2C address and Wire interface. The function returns true if the
    device is successfully initialized.
*/
/**************************************************************************/
bool Adafruit_HUSB238::begin(uint8_t addr, TwoWire *theWire) {
  if (i2c_dev) {
    delete i2c_dev;
  }
  i2c_dev = new Adafruit_I2CDevice(addr, theWire);

  /* Try to instantiate the I2C device. */
  if (!i2c_dev->begin()) {
    return false;
  }

  return true;
}

/**************************************************************************/
/*!
    @brief  Reads the CC status from the HUSB238 device.
    @return The CC status as a boolean value.
    @details This function reads the 7th bit of the HUSB238_PD_STATUS1 register to get the
    CC status. It returns true if the CC status bit is set.
*/
/**************************************************************************/
bool Adafruit_HUSB238::getCCStatus() {
  Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 7);  // 7th bit
  return ccBit.read();
}

// ... (continue with the rest of the functions in a similar manner)
