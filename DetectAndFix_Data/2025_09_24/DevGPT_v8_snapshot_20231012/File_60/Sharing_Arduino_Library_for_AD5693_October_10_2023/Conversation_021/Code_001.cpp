#include <Wire.h> // Include the Wire library for I2C communication

// Assuming you have a pointer to an Adafruit_I2CDevice object named i2c_dev

/**
 * @brief Initialize the AD5693 chip for communication.
 * 
 * This function initializes the I2C interface and sets up the AD5693 chip
 * for communication. It should be called in the setup() function of your
 * Arduino sketch.
 * 
 * @param addr The I2C address of the AD5693 chip.
 * @param wire The I2C interface to use (usually Wire).
 * @return true if the initialization was successful, false otherwise.
 */
bool begin(uint8_t addr, TwoWire &wire = Wire) {
  // Initialize the I2C interface
  wire.begin();

  // Create and initialize the I2C device object
  i2c_dev = new Adafruit_I2CDevice(addr, &wire);

  // Try to initialize the device and return the result
  return i2c_dev->begin();
}
