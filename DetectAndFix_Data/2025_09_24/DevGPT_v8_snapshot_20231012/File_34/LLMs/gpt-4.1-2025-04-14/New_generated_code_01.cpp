bool Adafruit_Debounce::read() {
    if (_pin >= 0) {
        return digitalRead(_pin) == _polarity;
    }
    return _buttonState;
}