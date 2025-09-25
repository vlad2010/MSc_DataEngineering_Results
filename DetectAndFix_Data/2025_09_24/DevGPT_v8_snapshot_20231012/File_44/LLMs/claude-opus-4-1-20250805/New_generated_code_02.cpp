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
    @brief  Instantiates a new HUSB238 class.
    @details This constructor initializes a new instance of the HUSB238 class.
*/
/**************************************************************************/
Adafruit_HUSB238::Adafruit_HUSB238() : i2c_dev(nullptr) {}

/**************************************************************************/
/*!
    @brief  Destructor for HUSB238 class.
    @details Cleans up dynamically allocated memory.
*/
/**************************************************************************/
Adafruit_HUSB238::~Adafruit_HUSB238() {
  if (i2c_dev) {
    delete i2c_dev;
    i2c_dev = nullptr;
  }
}

/**************************************************************************/
/*!
    @brief  Sets up the I2C connection and tests that the sensor was found.
    @param addr The 7-bit I2C address of the HUSB238.
    @param theWire Pointer to an I2C device we'll use to communicate; default is Wire.
    @return true if sensor was found, otherwise false.
    @details This function initializes the I2C communication with the HUSB238 device.
    It uses the provided I2C address and Wire interface. The function returns true if the
    device is successfully initialized.
*/
/**************************************************************************/
bool Adafruit_HUSB238::begin(uint8_t addr, TwoWire *theWire) {
  // Input validation
  if (theWire == nullptr) {
    return false;
  }
  
  // Validate I2C address range (7-bit address: 0x00-0x7F)
  if (addr > 0x7F) {
    return false;
  }
  
  // Clean up any existing device
  if (i2c_dev) {
    delete i2c_dev;
    i2c_dev = nullptr;
  }
  
  // Allocate new device with exception handling
  try {
    i2c_dev = new Adafruit_I2CDevice(addr, theWire);
  } catch (const std::bad_alloc&) {
    i2c_dev = nullptr;
    return false;
  }

  /* Try to instantiate the I2C device. */
  if (!i2c_dev->begin()) {
    delete i2c_dev;
    i2c_dev = nullptr;
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
  // Null pointer check
  if (!i2c_dev) {
    return false;
  }
  
  try {
    Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 7);  // 7th bit
    return ccBit.read();
  } catch (...) {
    // Handle any exceptions from I2C operations
    return false;
  }
}

/**************************************************************************/
/*!
    @brief  Reads the attachment status from the HUSB238 device.
    @return The attachment status as a boolean value.
    @details This function reads the 6th bit of the HUSB238_PD_STATUS1 register to get the
    attachment status. It returns true if the attachment status bit is set.
*/
/**************************************************************************/
bool Adafruit_HUSB238::getAttached() {
  // Null pointer check
  if (!i2c_dev) {
    return false;
  }
  
  try {
    Adafruit_I2CRegister attachRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit attachBit = Adafruit_I2CRegisterBit(&attachRegister, 6);  // 6th bit
    return attachBit.read();
  } catch (...) {
    // Handle any exceptions from I2C operations
    return false;
  }
}

// Additional helper method to check if device is initialized
bool Adafruit_HUSB238::isInitialized() const {
  return (i2c_dev != nullptr);
}

// Copy constructor (deleted to prevent shallow copies)
Adafruit_HUSB238::Adafruit_HUSB238(const Adafruit_HUSB238&) = delete;

// Copy assignment operator (deleted to prevent shallow copies)
Adafruit_HUSB238& Adafruit_HUSB238::operator=(const Adafruit_HUSB238&) = delete;

// Move constructor
Adafruit_HUSB238::Adafruit_HUSB238(Adafruit_HUSB238&& other) noexcept : i2c_dev(other.i2c_dev) {
  other.i2c_dev = nullptr;
}

// Move assignment operator
Adafruit_HUSB238& Adafruit_HUSB238::operator=(Adafruit_HUSB238&& other) noexcept {
  if (this != &other) {
    if (i2c_dev) {
      delete i2c_dev;
    }
    i2c_dev = other.i2c_dev;
    other.i2c_dev = nullptr;
  }
  return *this;
}