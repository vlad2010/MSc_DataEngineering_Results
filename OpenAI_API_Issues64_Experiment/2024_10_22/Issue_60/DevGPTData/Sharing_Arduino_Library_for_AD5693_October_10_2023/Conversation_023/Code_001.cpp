#ifndef ADAFRUIT_AD5693_H
#define ADAFRUIT_AD5693_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BusIO_Register.h>

/**
 * @brief Enum for AD5693 commands.
 */
typedef enum {
  /// No operation command.
  NOP = 0x00,
  /// Write to the input register.
  WRITE_INPUT = 0x10,
  /// Update the DAC register.
  UPDATE_DAC = 0x20,
  /// Write to the input register and update the DAC register.
  WRITE_DAC_AND_INPUT = 0x30,
  /// Write to the control register.
  WRITE_CONTROL = 0x40
} ad5693_commands;

/**
 * @brief Enum for AD569
