/**
 * @brief Update the button states based on the current pin reading.
 * 
 * This function updates the button states based on the result of digitalRead().
 */
void Adafruit_Debounce::update() {
    if (_pin >= 0) {
        _lastButtonState = _buttonState;
        _buttonState = digitalRead(_pin);
    }
}

/**
 * @brief Update the button states with a given state.
 * 
 * @param bit The state to set the button.
 */
void Adafruit_Debounce::update(bool bit) {
    _lastButtonState = _buttonState;
    _buttonState = bit;
}
