#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include "Arduino.h"

class Adafruit_Debounce {
  public:
    Adafruit_Debounce(int16_t pin, bool polarity = LOW);
    void begin();
    bool read();
    void update(bool bit);
    bool isPressed();
    bool isReleased();
    bool justPressed();
    bool justReleased();

  private:
    int16_t _pin;
    bool _polarity;
    uint32_t _lastDebounceTime;
    bool _buttonState; 
    bool _lastButtonState; 

    bool isValidPin(int16_t pin);
};

// Implementation of the Adafruit_Debounce class

Adafruit_Debounce::Adafruit_Debounce(int16_t pin, bool polarity)
  : _pin(pin), _polarity(polarity), _lastDebounceTime(0), _buttonState(false), _lastButtonState(false) 
{
    if (!isValidPin(pin)) {
        // Handle invalid pin situation, e.g., log an error, set default, throw exception, etc.
    }
}

bool Adafruit_Debounce::isValidPin(int16_t pin) {
    // Assuming the platform has digital pins ranging from 0 to N (e.g., 0 to 13 on Arduino Uno)
    // This function should be adjusted based on the specific platform
    return pin >= 0 && pin <= MAX_PIN_NUMBER; // Replace MAX_PIN_NUMBER with actual value based on platform
}

// Further method implementations go here...

#endif // ADAFRUIT_DEBOUNCE_H