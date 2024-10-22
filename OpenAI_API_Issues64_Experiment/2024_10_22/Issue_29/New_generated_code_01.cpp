bool Adafruit_HUSB238::get5VContractV() {
  if (i2c_dev == nullptr) {
    // Log error or handle the case where i2c_dev is not initialized properly
    return false; // Or throw an exception, depending on the context
  }

  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  
  // Assuming read will also handle the I2C communication status
  if (!pdRegister.init()) { // Hypothetical function to initialize/register
    // Handle the error if the I2C register cannot be initialized
    return false;
  }

  Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit

  bool result;
  bool readSuccess = contractVBit.read(&result);

  if (!readSuccess) {
    // Handle the error if read fails
    return false;
  }

  return result;
}