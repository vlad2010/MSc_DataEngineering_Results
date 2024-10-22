#include "Adafruit_Debounce.h"

const int16_t MIN_PIN = 0; // Modify according to specific hardware limits
const int16_t MAX_PIN = 255; // Modify according to specific hardware limits

/**
 * @brief Construct a new Adafruit_Debounce::Adafruit_Debounce object.
 * 
 * @param pin The pin number associated with the button.
 * @param polarity The polarity to determine the pressed state (default is LOW).
 */
Adafruit_Debounce::Adafruit_Debounce(int16_t pin, bool polarity) {
    if (pin < MIN_PIN || pin > MAX_PIN) {
        // Handle invalid pin
        handleError("Invalid pin number");
        _pin = -1;
    } else {
        _pin = pin;
    }

    if (polarity != LOW && polarity != HIGH) {
        // Handle invalid polarity
        handleError("Invalid polarity value");
        _polarity = LOW; // Default to LOW
    } else {
        _polarity = polarity;
    }

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
    if (_pin >= 0) {
        _buttonState = digitalRead(_pin);
    } else {
        _buttonState = initialState;
    }
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
        int readResult = digitalRead(_pin);
        if (readResult == -1) { // Assuming -1 represents error in this context
            handleError("Invalid read");
            return _buttonState;
        }
        return readResult;
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
        int readResult = digitalRead(_pin);
        if (readResult != -1) {
            _buttonState = readResult;
        }
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

/**
 * @brief Handle error message.
 * 
 * @param message Error message to display or log
 */
void Adafruit_Debounce::handleError(const char* message) {
    // Implement appropriate error handling (e.g., log message, notify user, etc.)
    Serial.println(message);
}