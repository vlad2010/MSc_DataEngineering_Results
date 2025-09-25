enum HUSB238_ResponseCodes {
  RESPONSE_CODE_0 = 0,
  RESPONSE_CODE_1,
  RESPONSE_CODE_2,
  RESPONSE_CODE_3,
  RESPONSE_CODE_4,
  RESPONSE_CODE_5,
  RESPONSE_CODE_6,
  RESPONSE_CODE_7,
  // ... add other codes as needed
  RESPONSE_CODE_INVALID = 255 // Use as an error code
};

#define MAX_RESPONSE_CODE RESPONSE_CODE_7 // Adjust as needed

HUSB238_ResponseCodes Adafruit_HUSB238::getPDResponse() {
  Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
  Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);  // Bits 3-5

  int value = pdBits.read();
  if (value < 0 || value > MAX_RESPONSE_CODE) {
    // Handle error: log, return error code, etc.
    return RESPONSE_CODE_INVALID;
  }
  return static_cast<HUSB238_ResponseCodes>(value);
}