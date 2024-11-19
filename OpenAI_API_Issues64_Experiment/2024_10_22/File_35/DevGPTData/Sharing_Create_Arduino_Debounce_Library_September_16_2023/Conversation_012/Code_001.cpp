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
        _buttonState = digitalRead(_pin) == _polarity;
        _lastButtonState = _buttonState;
    }
}
