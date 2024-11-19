#include "Adafruit_Debounce.h"

/**
 * @brief Construct a new Adafruit_Debounce::Adafruit_Debounce object.
 * 
 * @param pin The pin number associated with the button.
 * @param polarity The polarity to determine the pressed state (default is LOW).
 */
Adafruit_Debounce::Adafruit_Debounce(int16_t pin, bool polarity) {
    _pin = pin;
    _polarity = polarity;
    _buttonState = !_polarity;
    _lastButtonState = !_polarity;
}

/**
 * @brief Initialize the debounce object.
 * 
 * This function sets the pin mode based on the polarity and initializes
 * the button states.
 */
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
        
        _buttonState = digitalRead(_pin);
        _lastButtonState = _buttonState;
    }
}

/**
 * @brief Initialize the debounce object with a given initial state.
 * 
 * @param initialState The initial state to set the button states.
 */
void Adafruit_Debounce::begin(bool initialState) {
    begin();
    _buttonState = initialState;
    _lastButtonState = initialState;
}

/**
 * @brief Read the current state of the button.
 * 
 * @return true If the button state matches the polarity.
 * @return false Otherwise.
 */
bool Adafruit_Debounce::read() {
    if (_pin >= 0) {
        return digitalRead(_pin);
    }
    return _buttonState;
}

/**
 * @brief Update the button states.
 * 
 * @param bit The state to set the button if pin is invalid.
 */
void Adafruit_Debounce::update(bool bit) {
    _lastButtonState = _buttonState;
    
    if (_pin >= 0) {
        _buttonState = digitalRead(_pin);
    } else {
        _buttonState = bit;
    }
}

/**
 * @brief Check if the button is currently pressed.
 * 
 * @return true If the button is pressed.
 * @return false Otherwise.
 */
bool Adafruit_Debounce::isPressed() {
    return _buttonState == _polarity;
}

/**
 * @brief Check if the button is currently released.
 * 
 * @return true If the button is released.
 * @return false Otherwise.
 */
bool Adafruit_Debounce::isReleased() {
    return _buttonState != _polarity;
}

/**
 * @brief Check if the button was just pressed.
 * 
 * @return true If the button was just pressed.
 * @return false Otherwise.
 */
bool Adafruit_Debounce::justPressed() {
    return (_buttonState != _lastButtonState) && (_buttonState == _polarity);
}

/**
 * @brief Check if the button was just released.
 * 
 * @return true If the button was just released.
 * @return false Otherwise.
 */
bool Adafruit_Debounce::justReleased() {
    return (_buttonState != _lastButtonState) && (_buttonState != _polarity);
}
