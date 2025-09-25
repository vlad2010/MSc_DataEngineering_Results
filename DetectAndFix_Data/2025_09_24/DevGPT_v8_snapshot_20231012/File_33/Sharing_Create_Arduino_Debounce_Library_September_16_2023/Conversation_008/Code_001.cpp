void Adafruit_Debounce::begin() {
    if (_pin >= 0) {
        if (_polarity == LOW) {
            pinMode(_pin, INPUT_PULLUP);
        } 
        #ifdef INPUT_PULLDOWN
        else {
            pinMode(_pin, INPUT_PULLDOWN);
        }
        #endif
    }
}
