/**
 * @brief Set the operating mode, reference, and gain for the AD5693 chip.
 * 
 * This function writes to the control register of the AD5693 chip to set
 * the operating mode, enable or disable the reference, and set the gain.
 * It uses the I2C interface for communication and formats the data according
 * to the chip's specifications.
 * 
 * @param newmode The new operating mode to set (from ad5693_operating_modes enum).
 * @param enable_ref Whether to enable the reference (true) or disable it (false).
 * @param gain2x Whether to set the gain to 2x (true) or 1x (false).
 * @return true if the operation was successful, false otherwise.
 */
bool setMode(ad5693_operating_modes newmode, bool enable_ref, bool gain2x) {
  // Prepare the command byte
  uint8_t command = ad5693_commands::WRITE_CONTROL; // Assuming you've defined the enum as ad5693_commands

  // Prepare the high and low data bytes
  uint16_t data = 0x0000;
  data |= ((uint16_t)newmode << 13);  // Set D14 and D13 for the operating mode
  data |= ((uint16_t)enable_ref << 12); // Set D12 for enable_ref
  data |= ((uint16_t)gain2x << 11);  // Set D11 for the gain

  uint8_t highByte = (data >> 8) & 0xFF;
  uint8_t lowByte = data & 0xFF;

  // Combine the command and data into a single 3-byte buffer
  uint8_t buffer[3] = {command, highByte, lowByte};

  // Write the 3-byte buffer to the I2C device and return the result
  return i2c_dev->write(buffer, 3);
}
