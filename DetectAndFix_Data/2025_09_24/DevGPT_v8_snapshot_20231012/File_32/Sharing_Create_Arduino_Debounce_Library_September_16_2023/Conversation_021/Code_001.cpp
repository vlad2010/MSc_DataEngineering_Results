Adafruit_Debounce::Adafruit_Debounce(int16_t pin, bool polarity) {
    _pin = pin;
    _polarity = polarity;
    _buttonState = !_polarity;
    _lastButtonState = !_polarity;
}
