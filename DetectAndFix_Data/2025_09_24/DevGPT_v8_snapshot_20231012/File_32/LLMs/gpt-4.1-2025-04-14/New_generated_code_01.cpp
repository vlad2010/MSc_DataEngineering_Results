#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include "Arduino.h"

// Define the valid range for Arduino pins (example: 0-53 for Arduino Mega)
#define MIN_PIN_NUMBER 0
#define MAX_PIN_NUMBER 53

class Adafruit_Debounce {
  public:
    // Use uint8_t for pin to prevent negative values
    Adafruit_Debounce(uint8_t pin, bool polarity = LOW)
      : _pin(pin), _polarity(polarity), _lastDebounceTime(0),
        _buttonState(false), _lastButtonState(false)
    {
      // Validate pin number at runtime
      if (_pin < MIN_PIN_NUMBER || _pin > MAX_PIN_NUMBER) {
        // Handle invalid pin (could set to a safe default, or halt)
        // For demonstration, set to a safe default (e.g., 0)
        _pin = 0;
        // Optionally, log error or halt execution
        // Serial.println("Invalid pin number, defaulting to 0");
      }
    }

    void begin();
    bool read();
    void update(bool bit);
    bool isPressed();
    bool isReleased();
    bool justPressed();
    bool justReleased();

  private:
    uint8_t _pin;                // Pin number (now unsigned)
    bool _polarity;
    uint32_t _lastDebounceTime;  // Always initialized
    bool _buttonState;           // Always initialized
    bool _lastButtonState;       // Always initialized
};

#endif // ADAFRUIT_DEBOUNCE_H