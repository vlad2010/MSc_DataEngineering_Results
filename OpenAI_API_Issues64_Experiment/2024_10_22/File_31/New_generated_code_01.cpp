bool Adafruit_HUSB238::isVoltageDetected(HUSB238_PDOSelection pd) {
  uint8_t registerAddress;
  
  // Determine the register address based on the PDO selection
  switch(pd) {
    case PDO_5V:
      registerAddress = HUSB238_SRC_PDO_5V;
      break;
    case PDO_9V:
      registerAddress = HUSB238_SRC_PDO_9V;
      break;
    case PDO_12V:
      registerAddress = HUSB238_SRC_PDO_12V;
      break;
    case PDO_15V:
      registerAddress = HUSB238_SRC_PDO_15V;
      break;
    case PDO_18V:
      registerAddress = HUSB238_SRC_PDO_18V;
      break;
    case PDO_20V:
      registerAddress = HUSB238_SRC_PDO_20V;
      break;
    default:
      // Log an error or throw exception to handle unexpected PDO
      std::cerr << "Error: Unknown PDO" << std::endl;
      return false;
  }
  
  // Create an Adafruit_I2CRegister object for the selected register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, registerAddress);
  
  // Check if pdoRegister was initialized correctly
  if (!pdoRegister.initialized()) { // Assume `initialized()` is a valid method or similar error-check
    std::cerr << "Error initializing I2C Register for address: " << (int)registerAddress << std::endl;
    return false;
  }
  
  // Create an Adafruit_I2CRegisterBit object for the 7th bit
  Adafruit_I2CRegisterBit pdoBit = Adafruit_I2CRegisterBit(&pdoRegister, 7);

  // Attempt to read and check for errors
  bool bitValue;
  if (!pdoBit.read(bitValue)) { // Modified to check/fetch return success
    std::cerr << "Error reading from PDO bit" << std::endl;
    return false;
  }

  return bitValue;
}