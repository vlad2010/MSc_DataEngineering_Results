#include <stdexcept> // For std::runtime_error

// Assuming that HUSB238_5VCurrentContract is an enum
enum HUSB238_5VCurrentContract {
    CONTRACT_0,
    CONTRACT_1,
    CONTRACT_2,
    CONTRACT_3,
    // Add more contracts if there are any.
    // Ensure this list contains all possible valid contracts.
    MAX_CONTRACT // This should represent the count of contracts, not a valid one.
};

HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // Bottom two bits (0-1)
  
  int value = contractABits.read();

  // Validate the value before casting
  if (value >= 0 && value < MAX_CONTRACT) { // Check if value is within valid range
    return static_cast<HUSB238_5VCurrentContract>(value);
  } else {
    // Handle the error case: you might want to throw an exception, return a default value, or handle it another way
    throw std::runtime_error("Invalid contract value read from PD register.");
  }
}