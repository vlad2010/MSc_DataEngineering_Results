#ifndef ADAFRUIT_HUSB238_H
#define ADAFRUIT_HUSB238_H

#include "Arduino.h"

// Enum for current settings
typedef enum _husb_currents {
  // (Previous enum definitions for current settings)
} HUSB238_CurrentSetting;

// Enum for voltage settings
typedef enum _husb_voltages {
  // (Previous enum definitions for voltage settings)
} HUSB238_VoltageSetting;

// Enum for response codes
typedef enum _husb_response_codes {
  // (Previous enum definitions for response codes)
} HUSB238_ResponseCodes;

// Enum for 5V current contract
typedef enum _husb_5v_current_contract {
  // (Previous enum definitions for 5V current contract)
} HUSB238_5VCurrentContract;

// Enum for PDO selection
typedef enum _husb_pdo_selection {
  // (Previous enum definitions for PDO selection)
} HUSB238_PDOSelection;

class Adafruit_HUSB238 {
public:
  // Constructor
  Adafruit_HUSB238();

  // Initialization
  bool begin();

  // Methods for setting and getting current, voltage, etc.
  bool setCurrent(HUSB238_CurrentSetting setting);
  bool setVoltage(HUSB238_VoltageSetting setting);
  HUSB238_CurrentSetting getCurrent();
  HUSB238_VoltageSetting getVoltage();

  // Additional methods can be added here
  
private:
  // Private variables and methods
};

#endif // ADAFRUIT_HUSB238_H
