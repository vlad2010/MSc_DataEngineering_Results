#define NUM_DIGITAL_PINS 32 // Replace with actual number for your hardware

class Adafruit_Debounce {
public:
    Adafruit_Debounce(int pin, bool polarity)
        : _pin(pin), _polarity(polarity), _buttonState(false) {
        // Validate pin during construction
        if (_pin < 0 || _pin >= NUM_DIGITAL_PINS) {
            _pin = -1; // Mark as invalid
        }
    }

    bool read() {
        if (_pin >= 0 && _pin < NUM_DIGITAL_PINS) {
            return digitalRead(_pin) == _polarity;
        }
        // Optionally, log or handle invalid pin access here
        return _buttonState;
    }

private:
    int _pin;
    bool _polarity;
    bool _buttonState;
};