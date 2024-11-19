#ifndef ADAFRUIT_AD5693_H
#define ADAFRUIT_AD5693_H

#include <Adafruit_BusIO_Register.h>
#include <Wire.h>

class Adafruit_AD5693 {
public:
  Adafruit_AD5693();
  bool begin(uint8_t i2c_addr = AD5693_I2C_ADDRESS, TwoWire *theWire = &Wire);
  
  // Setter and Getter Functions
  void setVoltage(uint16_t voltage);
  uint16_t getVoltage();
  
  // Other functions based on the datasheet
  
private:
  Adafruit_BusIO_Register *i2c_dev;
};

#endif // ADAFRUIT_AD5693_H
