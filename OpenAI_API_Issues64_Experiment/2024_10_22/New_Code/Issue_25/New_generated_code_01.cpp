HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);

  // Create an Adafruit_I2CRegisterBits object for the bottom two bits (0-1) of PD_STATUS0
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // 2 bits starting from the 0th bit

  // Attempt to read the bits
  int contractValue = contractABits.read();
  
  // Validate that the read value is within expected range
  if (contractValue < 0 || contractValue > MAX_VALID_CONTRACT_VALUE) {
    // Handle error (log it, throw an exception, return a default value, etc.)
    // Custom handling code here, e.g., throw std::runtime_error("Invalid contract value");
  }

  // Safely cast and return the value, assuming contractValue is validated
  return static_cast<HUSB238_5VCurrentContract>(contractValue);
}