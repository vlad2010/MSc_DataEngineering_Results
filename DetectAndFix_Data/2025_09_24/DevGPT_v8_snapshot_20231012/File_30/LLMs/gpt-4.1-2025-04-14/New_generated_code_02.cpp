enum class HUSB238_5VCurrentContract {
  Contract0 = 0,
  Contract1 = 1,
  Contract2 = 2,
  Contract3 = 3,
  Unknown   = 255 // Use a value outside the valid range for error signaling
};

HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // Bottom two bits (0-1)
  int value = contractABits.read();

  // Check for I2C read error (assuming read() returns -1 on error)
  if (value < 0) {
    // Handle error, e.g., log or return a special value
    return HUSB238_5VCurrentContract::Unknown;
  }

  // Validate value is within expected enum range
  if (value >= static_cast<int>(HUSB238_5VCurrentContract::Contract0) &&
      value <= static_cast<int>(HUSB238_5VCurrentContract::Contract3)) {
    return static_cast<HUSB238_5VCurrentContract>(value);
  } else {
    // Value out of range, handle as error
    return HUSB238_5VCurrentContract::Unknown;
  }
}