bool Adafruit_Debounce::isPressed() {
    return _buttonState == _polarity;
}

bool Adafruit_Debounce::isReleased() {
    return _buttonState != _polarity;
}
