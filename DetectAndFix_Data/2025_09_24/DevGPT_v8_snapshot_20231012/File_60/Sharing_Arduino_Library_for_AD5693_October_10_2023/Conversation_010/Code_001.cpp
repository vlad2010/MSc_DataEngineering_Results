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

  // Write the command byte to the I2C device and return the result
  return i2c_dev->write(&command, 1);
}
