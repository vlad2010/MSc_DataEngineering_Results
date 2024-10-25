HUSB238_VoltageSetting Adafruit_HUSB238::getPDSrcVoltage() {
  // Create an Adafruit_I2CRegister object for the HUSB238_PD_STATUS0 register
  Adafruit_I2CRegister pd0Register = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);

  // Create an Adafruit_I2CRegisterBits object for bits 4-7 of HUSB238_PD_STATUS0
  Adafruit_I2CRegisterBits srcVoltageBits = Adafruit_I2CRegisterBits(&pd0Register, 4, 4);  // Bits 4-7

  // Read and return the bits, cast to HUSB238_VoltageSetting
  return (HUSB238_VoltageSetting) srcVoltageBits.read();
}
