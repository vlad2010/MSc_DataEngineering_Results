bool Adafruit_HUSB238::isVoltageDetected(HUSB238_PDOSelection pd) {
  // Create an Adafruit_I2CRegister object for the relevant register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, SOME_REGISTER);  // Replace SOME_REGISTER with the actual register
  
  // Create an Adafruit_I2CRegisterBits object for the relevant bits
  // The bit range will depend on the specific PDO selection
  Adafruit_I2CRegisterBits pdoBits = Adafruit_I2CRegisterBits(&pdoRegister, NUM_BITS, START_BIT);  // Replace NUM_BITS and START_BIT
  
  // Read the bits and compare against the PDO selection
  uint8_t detectedPdo = pdoBits.read();
  
  // Return true if the detected PDO matches the argument, false otherwise
  return (detectedPdo == static_cast<uint8_t>(pd));
}
