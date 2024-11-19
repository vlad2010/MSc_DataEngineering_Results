void Adafruit_HUSB238::selectPDO(HUSB238_PDOSelection pd) {
  // Create an Adafruit_I2CRegister object for the PDO_SELECT register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PDO_SELECT);

  // Create an Adafruit_I2CRegisterBits object for bits 4-7 of PDO_SELECT
  Adafruit_I2CRegisterBits pdoBits = Adafruit_I2CRegisterBits(&pdoRegister, 4, 4);  // Bits 4-7

  // Set the bits based on the provided HUSB238_PDOSelection value
  pdoBits.write(static_cast<uint8_t>(pd));
}
