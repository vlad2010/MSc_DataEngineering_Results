uint8_t Adafruit_HUSB238::getPDResponse() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);  // Removed default args
  
  // Create an Adafruit_I2CRegisterBits object for bits 3-5 of PD_STATUS0
  Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);  // 3 bits starting from the 3rd bit
  
  // Read and return the bits
  return pdBits.read();
}
