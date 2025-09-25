bool Adafruit_HUSB238::isVoltageDetected(HUSB238_PDOSelection pd) {
  // CWE-457 Fix: Initialize variable
  uint8_t registerAddress = 0;
  
  // CWE-476 Fix: Null pointer check
  if (i2c_dev == nullptr) {
    // Log error or handle appropriately
    return false;
  }
  
  // CWE-835 Fix: Use lookup table for better maintainability and security
  static const struct {
    HUSB238_PDOSelection pdo;
    uint8_t address;
  } pdoMapping[] = {
    {PDO_5V,  HUSB238_SRC_PDO_5V},
    {PDO_9V,  HUSB238_SRC_PDO_9V},
    {PDO_12V, HUSB238_SRC_PDO_12V},
    {PDO_15V, HUSB238_SRC_PDO_15V},
    {PDO_18V, HUSB238_SRC_PDO_18V},
    {PDO_20V, HUSB238_SRC_PDO_20V}
  };
  
  bool found = false;
  for (size_t i = 0; i < sizeof(pdoMapping)/sizeof(pdoMapping[0]); i++) {
    if (pdoMapping[i].pdo == pd) {
      registerAddress = pdoMapping[i].address;
      found = true;
      break;
    }
  }
  
  if (!found) {
    return false;  // Unknown PDO
  }
  
  // CWE-703 Fix: Add try-catch or error handling mechanism
  try {
    // Create objects with proper scope management
    Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, registerAddress);
    
    // Create bit accessor for the 7th bit
    Adafruit_I2CRegisterBit pdoBit = Adafruit_I2CRegisterBit(&pdoRegister, 7);
    
    // Read and return the bit with potential error handling
    bool result = pdoBit.read();
    
    // CWE-401 Fix: Objects will be properly destroyed when going out of scope
    // If the Adafruit libraries require explicit cleanup, add it here
    
    return result;
  } catch (...) {
    // CWE-703 Fix: Handle any exceptions from I2C operations
    // Log the error appropriately
    return false;
  }
}

// Alternative implementation with better error handling (if exceptions aren't available):
bool Adafruit_HUSB238::isVoltageDetectedSafe(HUSB238_PDOSelection pd, bool* success) {
  // Initialize success flag
  if (success != nullptr) {
    *success = false;
  }
  
  // Initialize variable
  uint8_t registerAddress = 0;
  
  // Null pointer check
  if (i2c_dev == nullptr) {
    return false;
  }
  
  // Use lookup table
  static const struct {
    HUSB238_PDOSelection pdo;
    uint8_t address;
  } pdoMapping[] = {
    {PDO_5V,  HUSB238_SRC_PDO_5V},
    {PDO_9V,  HUSB238_SRC_PDO_9V},
    {PDO_12V, HUSB238_SRC_PDO_12V},
    {PDO_15V, HUSB238_SRC_PDO_15V},
    {PDO_18V, HUSB238_SRC_PDO_18V},
    {PDO_20V, HUSB238_SRC_PDO_20V}
  };
  
  bool found = false;
  for (size_t i = 0; i < sizeof(pdoMapping)/sizeof(pdoMapping[0]); i++) {
    if (pdoMapping[i].pdo == pd) {
      registerAddress = pdoMapping[i].address;
      found = true;
      break;
    }
  }
  
  if (!found) {
    return false;
  }
  
  // Create objects with proper scope management
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, registerAddress);
  Adafruit_I2CRegisterBit pdoBit = Adafruit_I2CRegisterBit(&pdoRegister, 7);
  
  // Read the bit
  bool result = pdoBit.read();
  
  // Set success flag if provided
  if (success != nullptr) {
    *success = true;
  }
  
  return result;
}