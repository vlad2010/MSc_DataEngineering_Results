bool Adafruit_HUSB238::getCCStatus() {
  try {
    Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    if (!ccRegister.isValid()) {
      throw std::logic_error("Invalid I2C register access.");
    }
    Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 6);  // Ensure 6th bit is valid

    // Check bit access validity
    if (ccBit.getBitPosition() >= ccRegister.size()) {
      throw std::out_of_range("Bit position exceeds register size.");
    }
    
    return ccBit.read();
  } catch (const std::exception &e) {
    // Handle error (log, return a safe default value, etc.)
    std::cerr << "Error accessing CC Status: " << e.what() << std::endl;
    return false; // Or a safe default
  }
}