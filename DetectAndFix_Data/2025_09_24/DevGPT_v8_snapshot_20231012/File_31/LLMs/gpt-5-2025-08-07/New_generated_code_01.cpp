// Assumes this method is a member of Adafruit_HUSB238 and that `i2c_dev` is an Adafruit_I2CDevice*.

bool Adafruit_HUSB238::isVoltageDetected(HUSB238_PDOSelection pd) {
  // Security/robustness: avoid magic numbers and check for null device
  constexpr uint8_t kPDOReadyBit  = 7;
  constexpr uint8_t kPDOReadyMask = static_cast<uint8_t>(1u << kPDOReadyBit);

  if (i2c_dev == nullptr) {
    // CWE-476 fix: Null pointer check
    // Optionally record/log an error code here
    return false;
  }

  // Map PDO selection to the correct register address
  uint8_t registerAddress = 0;
  switch (pd) {
    case PDO_5V:  registerAddress = HUSB238_SRC_PDO_5V;  break;
    case PDO_9V:  registerAddress = HUSB238_SRC_PDO_9V;  break;
    case PDO_12V: registerAddress = HUSB238_SRC_PDO_12V; break;
    case PDO_15V: registerAddress = HUSB238_SRC_PDO_15V; break;
    case PDO_18V: registerAddress = HUSB238_SRC_PDO_18V; break;
    case PDO_20V: registerAddress = HUSB238_SRC_PDO_20V; break;
    default:
      // Defensive input validation (unknown PDO)
      return false;
  }

  // Read the register value with explicit error handling
  uint8_t value = 0;
  // CWE-252/CWE-703 fix: Check I2C transaction result
  if (!i2c_dev->write_then_read(&registerAddress, 1, &value, 1)) {
    // Optionally record/log an error code here
    return false; // False indicates "not detected OR read error"
  }

  // Return whether the PDO-ready bit is set
  return (value & kPDOReadyMask) != 0;
}