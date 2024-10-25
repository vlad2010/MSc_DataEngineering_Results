#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include "Arduino.h"

class Adafruit_Debounce {
  public:
    Adafruit_Debounce(int16_t pin);
    void begin();
    bool read();
    bool isPressed();
    bool isReleased();

  private:
    int16_t _pin;
    uint32_t _lastDebounceTime;
    bool _buttonState; 
    bool _lastButtonState; 
};

#endif // ADAFRUIT_DEBOUNCE_H
