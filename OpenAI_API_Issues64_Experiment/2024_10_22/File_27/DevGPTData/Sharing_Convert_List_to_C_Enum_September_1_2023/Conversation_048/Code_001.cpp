HUSB238_CurrentSetting Adafruit_HUSB238::currentDetected(HUSB238_PDOSelection pd) {
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
      return CURRENT_UNKNOWN;  // Unknown PDO, you can define this value in the enum if needed
  }
  
  // Create an Adafruit_I2CRegister object for the selected register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, registerAddress);
  
  // Create an Adafruit_I2CRegisterBits object for the bottom four bits (0-3)
  Adafruit_I2CRegisterBits currentBits = Adafruit_I2CRegisterBits(&pdoRegister, 4, 0);  // Bits 0-3
  
  // Read and return the bits, cast to HUSB238_CurrentSetting
  return (HUSB238_CurrentSetting) currentBits.read();
}
