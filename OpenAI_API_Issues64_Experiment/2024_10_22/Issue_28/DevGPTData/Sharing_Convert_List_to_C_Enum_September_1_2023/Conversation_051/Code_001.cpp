void Adafruit_HUSB238::requestPDO() {
  // Create an Adafruit_I2CRegister object for the GO_COMMAND register
  Adafruit_I2CRegister goCommandRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_GO_COMMAND);

  // Create an Adafruit_I2CRegisterBits object for bits 0-1 of GO_COMMAND
  Adafruit_I2CRegisterBits requestBits = Adafruit_I2CRegisterBits(&goCommandRegister, 1, 0);  // Bits 0-1

  // Write 0b00001 to the bits
  requestBits.write(0b00001);
}
