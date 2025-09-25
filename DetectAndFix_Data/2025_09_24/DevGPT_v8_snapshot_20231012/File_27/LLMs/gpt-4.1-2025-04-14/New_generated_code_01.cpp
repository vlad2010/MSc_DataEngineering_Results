bool Adafruit_HUSB238::getCCStatus(bool &success) {
  success = false;

  if (!i2c_dev) { // Validate input
    return false;
  }

  Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  if (!ccRegister.begin()) { // Hypothetical error check
    return false;
  }

  Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 6);  // 6th bit
  int bitValue = ccBit.read();
  if (bitValue < 0) { // Hypothetical: read() returns negative on error
    return false;
  }

  success = true;
  return bitValue != 0;
}