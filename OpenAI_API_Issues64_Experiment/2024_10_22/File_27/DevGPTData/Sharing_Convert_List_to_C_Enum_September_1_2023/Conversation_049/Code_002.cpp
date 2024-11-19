HUSB238_PDOSelection Adafruit_HUSB238::getSelectedPDO() {
  // Create an Adafruit_I2CRegister object for the PDO_SELECT register
  Adafruit_I2CRegister pdoRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PDO_SELECT);

  // Create an Adafruit_I2CRegisterBits object for bits 4-7 of PDO_SELECT
  Adafruit_I2CRegisterBits pdoBits = Adafruit_I2CRegisterBits(&pdoRegister, 4, 4);  // Bits 4-7

  // Read and return the bits, cast to HUSB238_PDOSelection
  return (HUSB238_PDOSelection) pdoBits.read();
}
