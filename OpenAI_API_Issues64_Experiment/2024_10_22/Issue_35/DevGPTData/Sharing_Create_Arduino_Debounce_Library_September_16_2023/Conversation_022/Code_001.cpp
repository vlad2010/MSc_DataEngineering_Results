bool Adafruit_Debounce::justPressed() {
    return (_buttonState != _lastButtonState) && (_buttonState == _polarity);
}

bool Adafruit_Debounce::justReleased() {
    return (_buttonState != _lastButtonState) && (_buttonState != _polarity);
}
