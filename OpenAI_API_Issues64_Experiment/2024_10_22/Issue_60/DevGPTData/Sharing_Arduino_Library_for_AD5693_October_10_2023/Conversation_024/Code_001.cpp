#ifndef ADAFRUIT_AD5693_H
#define ADAFRUIT_AD5693_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BusIO_Register.h>

/// Enum for AD5693 commands.
typedef enum {
  NOP = 0x00,               ///< No operation command.
  WRITE_INPUT = 0x10,       ///< Write to the input register.
  UPDATE_DAC = 0x20,        ///< Update the DAC register.
  WRITE_DAC_AND_INPUT = 0x30, ///< Write to the input register and update the DAC register.
  WRITE_CONTROL = 0x40      ///< Write to the control register.
} ad5693_commands;

/// Enum for AD5693 operating modes.
typedef enum {
  NORMAL_MODE = 0x00,             ///< Normal operating mode.
  OUTPUT_1K_IMPEDANCE = 0x01,     ///< 1k Ohm output impedance mode.
  OUTPUT_100K_IMPEDANCE = 0x02,   ///< 100k Ohm output impedance mode.
  OUTPUT_TRISTATE = 0x03          ///< Tristate output mode.
} ad5693_operating_modes;

/// Class for controlling the AD5693 chip.
class Adafruit_AD5693 {
public:
  Adafruit_AD5693(); ///< Constructor
  bool begin(uint8_t addr, TwoWire &wire = Wire); ///< Initialize the AD5693 chip for communication.
  bool writeDAC(uint16_t value); ///< Write a 16-bit value to the DAC register.
  bool updateDAC(void); ///< Update the DAC register from the input register.
  bool writeUpdateDAC(uint16_t value); ///< Write a 16-bit value to the input register and update the DAC register.
  bool reset(void); ///< Reset the AD5693 chip.
  bool setMode(ad5693_operating_modes newmode, bool enable_ref, bool gain2x); ///< Set the operating mode, reference, and gain for the AD5693 chip.

private:
  Adafruit_I2CDevice *i2c_dev; ///< Pointer to I2C device.
};

#endif // ADAFRUIT_AD5693_H
