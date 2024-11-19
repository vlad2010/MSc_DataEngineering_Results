#include "Adafruit_Debounce.h"

Adafruit_Debounce::Adafruit_Debounce(int16_t pin, bool polarity) {
    _pin = pin;
    _polarity = polarity;
    _buttonState = digitalRead(_pin);
    _lastButtonState = _buttonState;
}

void Adafruit_Debounce::begin() {
    if (_pin >= 0) {
        if (_polarity == LOW) {
            pinMode(_pin, INPUT_PULLUP);
        } else {
            #ifdef INPUT_PULLDOWN
            pinMode(_pin, INPUT_PULLDOWN);
            #else
            pinMode(_pin, INPUT);
            #endif
        }
        
        // Initialize the button states after setting the pin mode
        _buttonState = digitalRead(_pin);
        _lastButtonState = _buttonState;
    }
}

void Adafruit_Debounce::begin(bool initialState) {
    // Call the original begin() function
    begin();
    
    // Set the button states to the provided initialState value
    _buttonState = initialState;
    _lastButtonState = initialState;
}

bool Adafruit_Debounce::read() {
    if (_pin >= 0) {
        return digitalRead(_pin);
    }
    return _buttonState;
}

void Adafruit_Debounce::update(bool bit) {
    _lastButtonState = _buttonState;
    
    if (_pin >= 0) {
        _buttonState = digitalRead(_pin);
    } else {
        _buttonState = bit;
    }
}

bool Adafruit_Debounce::isPressed() {
    return _buttonState == _polarity;
}

bool Adafruit_Debounce::isReleased() {
    return _buttonState != _polarity;
}
