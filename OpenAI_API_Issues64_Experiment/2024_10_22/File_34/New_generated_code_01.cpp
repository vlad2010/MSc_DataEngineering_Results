#define MAX_PIN_NUM 13 // Assuming a maximum of 13 digital pins (common for an Arduino UNO)

bool Adafruit_Debounce::read() {
    if (_pin >= 0 && _pin <= MAX_PIN_NUM) {
        return digitalRead(_pin) == _polarity;
    }
    return _buttonState;
}