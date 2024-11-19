bool Adafruit_HUSB238::getCCStatus() {
  Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 7);  // 7th bit
  return ccBit.read();
}
