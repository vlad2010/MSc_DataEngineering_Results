#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include "Arduino.h"

class Adafruit_Debounce {
  public:
    Adafruit_Debounce(uint8_t pin, uint16_t debounceDelay = 50);
    void begin();
    bool read();
    bool isPressed();
    bool isReleased();

  private:
    uint8_t _pin;
    uint16_t _debounceDelay;
    uint32_t _lastDebounceTime;
    int _buttonState; 
    int _lastButtonState; 
};

#endif // ADAFRUIT_DEBOUNCE_H
