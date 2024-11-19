HUSB238_CurrentSetting Adafruit_HUSB238::getPDSrcCurrent() {
  // Create an Adafruit_I2CRegister object for the HUSB238_PD_STATUS0 register
  Adafruit_I2CRegister pd0Register = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);

  // Create an Adafruit_I2CRegisterBits object for the bottom four bits (0-3) of HUSB238_PD_STATUS0
  Adafruit_I2CRegisterBits srcCurrentBits = Adafruit_I2CRegisterBits(&pd0Register, 4, 0);  // Bits 0-3

  // Read and return the bits, cast to HUSB238_CurrentSetting
  return (HUSB238_CurrentSetting) srcCurrentBits.read();
}
