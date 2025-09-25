#include <Arduino.h>

class Adafruit_Debounce {
public:
  // Ensure members are initialized
  explicit Adafruit_Debounce(int pin = -1, bool polarityActiveHigh = true, bool initialState = false)
  : _pin(pin), _polarity(polarityActiveHigh), _buttonState(initialState) {}

  // Thread/ISR-safe setter if state can change from interrupts
  void setButtonState(bool state) {
    noInterrupts();
    _buttonState = state;
    interrupts();
  }

  void setPin(int pin) { _pin = pin; }
  void setPolarity(bool activeHigh) { _polarity = activeHigh; }

  bool read();

private:
  // Validate pin index; prefer NUM_DIGITAL_PINS when available
  static inline bool isValidDigitalPin(int pin) {
#if defined(NUM_DIGITAL_PINS)
    return (pin >= 0) && (pin < NUM_DIGITAL_PINS);
#else
    // Best-effort: we can only ensure non-negative if NUM_DIGITAL_PINS is not provided
    return (pin >= 0);
#endif
  }

  int _pin;                    // logical pin index
  bool _polarity;              // true -> active HIGH, false -> active LOW
  volatile bool _buttonState;  // may be updated from ISR/polling
};

bool Adafruit_Debounce::read() {
  if (isValidDigitalPin(_pin)) {
    // Avoid numeric truncation by validating, then explicitly casting
    const int val = digitalRead(static_cast<uint8_t>(_pin));
    return (val == (_polarity ? HIGH : LOW));
  }

  // Fallback for invalid pin: return a stable cached state without races
  bool state;
  noInterrupts();
  state = _buttonState;
  interrupts();
  return state;
}