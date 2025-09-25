/**
 * @brief Reset the AD5693 chip.
 * 
 * This function writes 0x8000 to the control register of the AD5693 chip
 * to perform a reset operation. It uses the I2C interface for communication
 * and formats the data according to the chip's specifications.
 * 
 * @return true if the reset operation was successful, false otherwise.
 */
bool reset(void) {
  // Prepare the command byte
  uint8_t command = ad5693_commands::WRITE_CONTROL; // Assuming you've defined the enum as ad5693_commands

  // Prepare the high and low data bytes for 0x8000
  uint8_t highByte = 0x80;
  uint8_t lowByte = 0x00;

  // Combine the command and data into a single 3-byte buffer
  uint8_t buffer[3] = {command, highByte, lowByte};

  // Write the 3-byte buffer to the I2C device and return the result
  return i2c_dev->write(buffer, 3);
}
