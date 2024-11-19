/**
 * @brief Write a 16-bit value to the input register and update the DAC register.
 * 
 * This function writes a 16-bit value to the input register and then updates
 * the DAC register of the AD5693 chip in a single operation. It uses the
 * I2C interface for communication and formats the data according to the
 * chip's specifications.
 * 
 * @param value The 16-bit value to write to the input register and update the DAC register.
 * @return true if the write and update operation was successful, false otherwise.
 */
bool writeUpdateDAC(uint16_t value) {
  // Prepare the command byte
  uint8_t command = ad5693_commands::WRITE_DAC_AND_INPUT; // Assuming you've defined the enum as ad5693_commands

  // Prepare the high and low data bytes
  uint8_t highByte = (value >> 8) & 0xFF;
  uint8_t lowByte = value & 0xFF;

  // Combine the command and data into a single 3-byte buffer
  uint8_t buffer[3] = {command, highByte, lowByte};

  // Write the 3-byte buffer to the I2C device and return the result
  return i2c_dev->write(buffer, 3);
}
