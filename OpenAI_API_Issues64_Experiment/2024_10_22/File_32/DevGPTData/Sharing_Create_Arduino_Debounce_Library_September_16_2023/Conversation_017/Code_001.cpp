void Adafruit_Debounce::update(bool bit) {
    _lastButtonState = _buttonState;
    
    if (_pin >= 0) {
        _buttonState = (digitalRead(_pin) == _polarity);
    } else {
        _buttonState = bit;
    }
}
