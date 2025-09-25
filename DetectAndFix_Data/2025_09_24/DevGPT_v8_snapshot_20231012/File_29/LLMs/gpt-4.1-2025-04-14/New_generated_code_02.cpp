#include <stdexcept> // For std::runtime_error

bool Adafruit_HUSB238::get5VContractV() {
  if (i2c_dev == nullptr) {
    // Defensive: Check for null pointer
    throw std::runtime_error("I2C device pointer is null");
  }

  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);

  // Optionally, check if pdRegister is valid (depends on library API)
  if (!pdRegister.isValid()) {
    throw std::runtime_error("Failed to initialize I2C register");
  }

  Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);  // 2nd bit

  // Optionally, check if contractVBit is valid (depends on library API)
  if (!contractVBit.isValid()) {
    throw std::runtime_error("Failed to initialize I2C register bit");
  }

  int result = contractVBit.read();
  if (result < 0) { // Assuming negative value indicates error
    throw std::runtime_error("Failed to read I2C register bit");
  }

  return result != 0;
}