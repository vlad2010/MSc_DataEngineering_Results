#include <stdexcept>
#include <limits>

/**************************************************************************/
/*!
    @brief  Reads the PD response from the HUSB238 device.
    @return The PD response as an HUSB238_ResponseCodes enum value.
    @throws std::runtime_error if I2C device is null or communication fails
    @throws std::out_of_range if the read value is outside valid enum range
    @details This function reads bits 3-5 of the HUSB238_PD_STATUS1 register to get the
    PD response. It returns the response as an HUSB238_ResponseCodes enum value.
*/
/**************************************************************************/
HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponse() {
  // CWE-476: Check for null pointer
  if (i2c_dev == nullptr) {
    throw std::runtime_error("I2C device not initialized");
  }
  
  try {
    Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);  // Bits 3-5
    
    uint8_t rawValue = pdBits.read();
    
    // CWE-704, CWE-20: Validate the value before casting
    // Assuming HUSB238_ResponseCodes has values 0-7 (3 bits)
    const uint8_t MAX_RESPONSE_CODE = 7;  // Adjust based on actual enum definition
    if (rawValue > MAX_RESPONSE_CODE) {
      throw std::out_of_range("Invalid PD response code: " + std::to_string(rawValue));
    }
    
    return static_cast<HUSB238_ResponseCodes>(rawValue);
  } catch (const std::exception& e) {
    // CWE-252, CWE-394: Handle I2C communication errors
    throw std::runtime_error(std::string("Failed to read PD response: ") + e.what());
  }
}

/**************************************************************************/
/*!
    @brief  Reads the 5V contract voltage status from the HUSB238 device.
    @return The 5V contract voltage status as a boolean value.
    @throws std::runtime_error if I2C device is null or communication fails
    @details This function reads the 2nd bit of the HUSB238_PD_STATUS1 register to get the
    5V contract voltage status. It returns true if the 5V contract voltage bit is set.
*/
/**************************************************************************/
bool Adafruit_HUSB238::get5VContractV() {
  // CWE-476: Check for null pointer
  if (i2c_dev == nullptr) {
    throw std::runtime_error("I2C device not initialized");
  }
  
  try {
    Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit
    
    // Boolean conversion is safe, but we still handle exceptions
    return contractVBit.read();
  } catch (const std::exception& e) {
    // CWE-252, CWE-394: Handle I2C communication errors
    throw std::runtime_error(std::string("Failed to read 5V contract voltage: ") + e.what());
  }
}

/**************************************************************************/
/*!
    @brief  Reads the 5V contract current from the HUSB238 device.
    @return The 5V contract current as an HUSB238_5VCurrentContract enum value.
    @throws std::runtime_error if I2C device is null or communication fails
    @throws std::out_of_range if the read value is outside valid enum range
    @details This function reads the bottom two bits (0-1) of the HUSB238_PD_STATUS1 register to get the
    5V contract current. It returns the current as an HUSB238_5VCurrentContract enum value.
*/
/**************************************************************************/
HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  // CWE-476: Check for null pointer
  if (i2c_dev == nullptr) {
    throw std::runtime_error("I2C device not initialized");
  }
  
  try {
    Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // Bottom two bits (0-1)
    
    uint8_t rawValue = contractABits.read();
    
    // CWE-704, CWE-20: Validate the value before casting
    // 2 bits can have values 0-3
    const uint8_t MAX_CURRENT_CONTRACT = 3;  // Adjust based on actual enum definition
    if (rawValue > MAX_CURRENT_CONTRACT) {
      throw std::out_of_range("Invalid 5V current contract: " + std::to_string(rawValue));
    }
    
    return static_cast<HUSB238_5VCurrentContract>(rawValue);
  } catch (const std::exception& e) {
    // CWE-252, CWE-394: Handle I2C communication errors
    throw std::runtime_error(std::string("Failed to read 5V contract current: ") + e.what());
  }
}

// Alternative approach using error codes instead of exceptions (for embedded systems)
// This would be more suitable for resource-constrained environments

enum class HUSB238_ErrorCode {
  SUCCESS = 0,
  NULL_DEVICE = -1,
  I2C_ERROR = -2,
  INVALID_VALUE = -3
};

struct HUSB238_Result {
  HUSB238_ErrorCode error;
  union {
    HUSB238_ResponseCodes response;
    HUSB238_5VCurrentContract current;
    bool boolValue;
    uint8_t rawValue;
  } value;
};

HUSB238_Result Adafruit_HUSB238::getPDResponseSafe() {
  HUSB238_Result result;
  
  // CWE-476: Check for null pointer
  if (i2c_dev == nullptr) {
    result.error = HUSB238_ErrorCode::NULL_DEVICE;
    return result;
  }
  
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);
  
  // Assuming read() returns a special value on error (e.g., 0xFF)
  uint8_t rawValue = pdBits.read();
  
  // Check for I2C error (implementation-specific)
  if (rawValue == 0xFF) {  // Adjust based on actual error indication
    result.error = HUSB238_ErrorCode::I2C_ERROR;
    return result;
  }
  
  // CWE-704, CWE-20: Validate the value before casting
  const uint8_t MAX_RESPONSE_CODE = 7;
  if (rawValue > MAX_RESPONSE_CODE) {
    result.error = HUSB238_ErrorCode::INVALID_VALUE;
    result.value.rawValue = rawValue;
    return result;
  }
  
  result.error = HUSB238_ErrorCode::SUCCESS;
  result.value.response = static_cast<HUSB238_ResponseCodes>(rawValue);
  return result;
}