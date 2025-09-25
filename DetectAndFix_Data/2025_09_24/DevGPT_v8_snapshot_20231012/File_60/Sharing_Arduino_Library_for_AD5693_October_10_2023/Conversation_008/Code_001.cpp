#include <Adafruit_BusIO_Register.h> // Include the Adafruit_BusIO library

// Assuming you have an Adafruit_I2CDevice object named i2c_dev

bool writeDAC(uint16_t value) {
  // Prepare the command and data bytes
  uint8_t command = ad5693_commands::WRITE_DAC_AND_INPUT; // Assuming you've defined the enum as ad5693_commands
  uint8_t highByte = (value >> 8) & 0xFF;
  uint8_t lowByte = value & 0xFF;

  // Combine the command and data into a single 3-byte buffer
  uint8_t buffer[3];
  buffer[0] = command;
  buffer[1] = highByte;
  buffer[2] = lowByte;

  // Write the buffer to the I2C device and return the result
  return i2c_dev->write(buffer, 3);
}
