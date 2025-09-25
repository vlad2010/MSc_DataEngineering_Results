void Adafruit_Debounce::update(bool bit) {
    if (_pin >= 0) {
        _lastButtonState = _buttonState;
        _buttonState = (digitalRead(_pin) == _polarity);
    } else {
        _lastButtonState = _buttonState;
        _buttonState = bit;
    }
}
