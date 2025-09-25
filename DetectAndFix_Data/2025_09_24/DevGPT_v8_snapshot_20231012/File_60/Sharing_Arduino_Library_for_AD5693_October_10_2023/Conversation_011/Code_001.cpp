/**
 * @brief Update the DAC register from the input register.
 * 
 * This function sends the UPDATE_DAC command to the AD5693 chip to update
 * the DAC register based on the value stored in the input register.
 * 
 * @return true if the update operation was successful, false otherwise.
 */
bool updateDAC(void) {
  // Prepare the command byte
  uint8_t command = ad5693_commands::UPDATE_DAC; // Assuming you've defined the enum as ad5693_commands

  // Prepare a 3-byte buffer: command byte followed by two 0x00 bytes
  uint8_t buffer[3] = {command, 0x00, 0x00};

  // Write the 3-byte buffer to the I2C device and return the result
  return i2c_dev->write(buffer, 3);
}
