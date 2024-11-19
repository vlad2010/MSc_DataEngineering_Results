bool Adafruit_HUSB238::getCCStatus() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);  // Replace with actual register and config
  
  // Create an Adafruit_I2CRegisterBit object for the 6th bit of PD_STATUS0, known as "CC_DIR"
  Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 6);  // 6th bit

  // Read the bit
  bool ccStatus = ccBit.read();

  return ccStatus;
}