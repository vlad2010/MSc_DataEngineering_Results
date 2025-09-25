#ifndef ADAFRUIT_AD5693_H
#define ADAFRUIT_AD5693_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BusIO_Register.h>

// Enum for AD5693 commands
typedef enum {
  NOP = 0x00,
  WRITE_INPUT = 0x10,
  UPDATE_DAC = 0x20,
  WRITE_DAC_AND_INPUT = 0x30,
  WRITE_CONTROL = 0x40
} ad5693_commands;

// Enum for AD5693 operating modes
typedef enum {
  NORMAL_MODE = 0x00,
  OUTPUT_1K_IMPEDANCE = 0x01,
  OUTPUT_100K_IMPEDANCE = 0x02,
  OUTPUT_TRISTATE = 0x03
} ad5693_operating_modes;

class Adafruit_AD5693 {
public:
  Adafruit_AD5693(); // Constructor
  bool begin(uint8_t addr, TwoWire &wire = Wire);
  bool writeDAC(uint16_t value);
  bool updateDAC(void);
  bool writeUpdateDAC(uint16_t value);
  bool reset(void);
  bool setMode(ad5693_operating_modes newmode, bool enable_ref, bool gain2x);

private:
  Adafruit_I2CDevice *i2c_dev; // Pointer to I2C device
};

#endif // ADAFRUIT_AD5693_H
