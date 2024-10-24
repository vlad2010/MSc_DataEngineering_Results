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
      return false;  // Unknown PDO
  }
  
  // Create an Adafruit_I2CRegister object for the selected register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, registerAddress);
  
  // Create an Adafruit_I2CRegisterBit object for the 7th bit
  Adafruit_I2CRegisterBit pdoBit = Adafruit_I2CRegisterBit(&pdoRegister, 7);  // 7th bit
  
  // Read and return the bit
  return pdoBit.read();
}
