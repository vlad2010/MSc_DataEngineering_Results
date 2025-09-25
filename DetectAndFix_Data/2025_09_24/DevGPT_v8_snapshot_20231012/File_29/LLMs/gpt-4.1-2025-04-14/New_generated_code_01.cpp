bool Adafruit_HUSB238::get5VContractV() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit
  return contractVBit.read();
}