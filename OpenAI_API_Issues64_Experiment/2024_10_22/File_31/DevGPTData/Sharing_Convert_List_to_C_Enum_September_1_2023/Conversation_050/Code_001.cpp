void Adafruit_HUSB238::reset() {
  // Create an Adafruit_I2CRegister object for the GO_COMMAND register
  Adafruit_I2CRegister goCommandRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_GO_COMMAND);

  // Create an Adafruit_I2CRegisterBits object for the bottom 5 bits (0-4) of GO_COMMAND
  Adafruit_I2CRegisterBits resetBits = Adafruit_I2CRegisterBits(&goCommandRegister, 5, 0);  // Bits 0-4

  // Write 0b10000 to the bottom 5 bits
  resetBits.write(0b10000);
}
