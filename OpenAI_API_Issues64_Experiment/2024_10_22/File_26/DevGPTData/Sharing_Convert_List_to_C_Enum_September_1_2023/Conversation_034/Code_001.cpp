bool Adafruit_HUSB238::get5VContractV() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);
  
  // Create an Adafruit_I2CRegisterBit object for the 2nd bit of PD_STATUS0
  Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit
  
  // Read and return the bit
  return contractVBit.read();
}
