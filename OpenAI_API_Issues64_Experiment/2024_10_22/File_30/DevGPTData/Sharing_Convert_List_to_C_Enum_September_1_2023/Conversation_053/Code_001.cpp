void Adafruit_HUSB238::getSourceCapabilities() {
  // Create an Adafruit_I2CRegister object for the GO_COMMAND register
  Adafruit_I2CRegister goCommandRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_GO_COMMAND);

  // Create an Adafruit_I2CRegisterBits object for the bottom 5 bits (0-4) of GO_COMMAND
  Adafruit_I2CRegisterBits capabilitiesBits = Adafruit_I2CRegisterBits(&goCommandRegister, 5, 0);  // Bits 0-4

  // Write 0b00100 to the bits
  capabilitiesBits.write(0b00100);
}
