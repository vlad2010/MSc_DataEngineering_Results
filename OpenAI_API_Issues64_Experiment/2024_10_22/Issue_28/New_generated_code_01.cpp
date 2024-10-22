enum HUSB238_ResponseCodes {
  RESPONSE_CODE_0 = 0,
  RESPONSE_CODE_1 = 1,
  RESPONSE_CODE_2 = 2,
  RESPONSE_CODE_3 = 3,
  RESPONSE_CODE_ERROR = -1  // Assume an error code
};

HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponse() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);  // Bits 3-5
  uint8_t rawValue = pdBits.read();

  // Validate the value to ensure it matches one of the known codes
  switch (rawValue) {
    case RESPONSE_CODE_0:
    case RESPONSE_CODE_1:
    case RESPONSE_CODE_2:
    case RESPONSE_CODE_3:
      return static_cast<HUSB238_ResponseCodes>(rawValue);
    default:
      return RESPONSE_CODE_ERROR;  // Return a special error code if an unexpected value is read
  }
}