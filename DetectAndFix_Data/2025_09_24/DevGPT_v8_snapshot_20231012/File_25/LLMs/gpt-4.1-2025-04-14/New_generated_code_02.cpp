enum HUSB238_5VCurrentContract {
  CONTRACT_0 = 0,
  CONTRACT_1 = 1,
  CONTRACT_2 = 2,
  CONTRACT_3 = 3,
  CONTRACT_INVALID = -1 // Add an explicit invalid value
};

HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);

  int value = contractABits.read();
  if (value < 0 || value > 3) { // Validate the value is within the enum range
    // Optionally log the error or handle it as needed
    return CONTRACT_INVALID;
  }
  return static_cast<HUSB238_5VCurrentContract>(value);
}