HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);
  
  // Create an Adafruit_I2CRegisterBits object for the bottom two bits (0-1) of PD_STATUS0
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // 2 bits starting from the 0th bit
  
  // Read and return the bits, cast to HUSB238_5VCurrentContract
  return (HUSB238_5VCurrentContract) contractABits.read();
}