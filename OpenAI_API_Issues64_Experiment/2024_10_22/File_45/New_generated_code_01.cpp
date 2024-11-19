/**************************************************************************/
/*!
    @brief  Reads the PD response from the HUSB238 device.
    @return The PD response as an HUSB238_ResponseCodes enum value.
    @details This function reads bits 3-5 of the HUSB238_PD_STATUS1 register to get the
    PD response. It returns the response as an HUSB238_ResponseCodes enum value.
*/
/**************************************************************************/
HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponse() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);  // Bits 3-5
  int value = pdBits.read();
  
  if (value < 0 || value >= MAX_HUSB238_ResponseCodes) {  // Assuming MAX_HUSB238_ResponseCodes is defined to the max value of enum
    // Handle error: either throw an exception, return a default value, or log an error
    // return HUSB238_ResponseCodes::Invalid;  // Assume an invalid enum value is defined
  }
  
  return static_cast<HUSB238_ResponseCodes>(value);
}

/**************************************************************************/
/*!
    @brief  Reads the 5V contract voltage status from the HUSB238 device.
    @return The 5V contract voltage status as a boolean value.
    @details This function reads the 2nd bit of the HUSB238_PD_STATUS1 register to get the
    5V contract voltage status. It returns true if the 5V contract voltage bit is set.
*/
/**************************************************************************/
bool Adafruit_HUSB238::get5VContractV() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit
  
  // If the library supports checking errors, wrap this to verify the read was successful.
  return contractVBit.read();
}

/**************************************************************************/
/*!
    @brief  Reads the 5V contract current from the HUSB238 device.
    @return The 5V contract current as an HUSB238_5VCurrentContract enum value.
    @details This function reads the bottom two bits (0-1) of the HUSB238_PD_STATUS1 register to get the
    5V contract current. It returns the current as an HUSB238_5VCurrentContract enum value.
*/
/**************************************************************************/
HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // Bottom two bits (0-1)
  int value = contractABits.read();
  
  if (value < 0 || value >= MAX_HUSB238_5VCurrentContract) {  // Assuming MAX_HUSB238_5VCurrentContract is defined
    // Handle error appropriately, similar to the method above.
    // return HUSB238_5VCurrentContract::Invalid;  // Assume an invalid enum value is defined
  }
  
  return static_cast<HUSB238_5VCurrentContract>(value);
}