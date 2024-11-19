bool Adafruit_HUSB238::getAttached() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister attachRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);  // Replace with actual register and config
  
  // Create an Adafruit_I2CRegisterBit object for the 6th bit of PD_STATUS0
  Adafruit_I2CRegisterBit attachBit = Adafruit_I2CRegisterBit(&attachRegister, 6);  // 6th bit

  // Read the bit
  bool attachStatus = attachBit.read();

  return attachStatus;
}
