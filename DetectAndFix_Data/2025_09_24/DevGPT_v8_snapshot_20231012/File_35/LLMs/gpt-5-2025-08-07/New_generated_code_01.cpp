#include <Arduino.h>

class Adafruit_Debounce {
public:
  // polarity: HIGH means "pressed when HIGH", LOW means "pressed when LOW"
  // debounceMs: time window for debouncing/glitch filtering
  explicit Adafruit_Debounce(int16_t pin, uint8_t polarity = LOW, uint16_t debounceMs = 20)
  : _pin(pin),
    _polarity((polarity == HIGH) ? HIGH : LOW),
    _buttonState(!_toBool(_polarity)),   // initialize opposite of "pressed"
    _lastButtonState(!_toBool(_polarity)),
    _rawState(!_toBool(_polarity)),
    _debounceStart(0),
    _debounceMs(debounceMs),
    _hasValidPin(false)
  {}

  // Initialize hardware, validate pin, configure pull resistors, and seed states.
  void begin() {
    _hasValidPin = pinIsValidStatic(_pin);
    if (_hasValidPin) {
      if (_polarity == LOW) {
        // Pressed when LOW => prefer internal pull-up to keep line stable otherwise
        pinMode(_pin, INPUT_PULLUP);
      } else {
        // Pressed when HIGH => prefer internal pulldown if supported
        #ifdef INPUT_PULLDOWN
          pinMode(_pin, INPUT_PULLDOWN);
        #else
          // Fallback to INPUT; requires external pulldown for robust behavior
          pinMode(_pin, INPUT);
        #endif
      }
      int val = digitalRead(_pin);
      _rawState = (val == HIGH);
      _buttonState = _rawState;
      _lastButtonState = _rawState;
    } else {
      // Disable hardware access when pin is invalid to avoid UB in core
      _pin = -1;
      // Keep current logical state; use software-supplied bit in update()
      _rawState = _buttonState;
    }
    _debounceStart = millis();
  }

  // Initialize with an explicit logical initial state (debounced and raw)
  void begin(bool initialState) {
    begin();
    _rawState = initialState;
    _buttonState = initialState;
    _lastButtonState = initialState;
    _debounceStart = millis();
  }

  // Periodically call to refresh debounced state.
  // If _pin is invalid, 'bit' provides the raw input.
  void update(bool bit = false) {
    bool newRaw = _hasValidPin ? (digitalRead(_pin) == HIGH) : bit;
    uint32_t now = millis();

    // Start/restart debounce timer on any raw change
    if (newRaw != _rawState) {
      _rawState = newRaw;
      _debounceStart = now;
    }

    // Commit change only after stable for debounce window
    if ((uint32_t)(now - _debounceStart) >= _debounceMs && _buttonState != _rawState) {
      _lastButtonState = _buttonState;
      _buttonState = _rawState;
    }
  }

  // Return the debounced logic level (true == HIGH)
  bool read() const {
    return _buttonState;
  }

  // Pressed if debounced state equals the configured polarity
  bool isPressed() const {
    return _buttonState == _toBool(_polarity);
  }

  bool isReleased() const {
    return _buttonState != _toBool(_polarity);
  }

  bool justPressed() const {
    return (_buttonState != _lastButtonState) && (_buttonState == _toBool(_polarity));
  }

  bool justReleased() const {
    return (_buttonState != _lastButtonState) && (_buttonState != _toBool(_polarity));
  }

  bool hasValidPin() const { return _hasValidPin; }

private:
  // Convert HIGH/LOW to bool (HIGH -> true, LOW -> false)
  static constexpr bool _toBool(uint8_t level) { return level == HIGH; }

  // Validate pin within platform limits to avoid UB in core
  static bool pinIsValidStatic(int16_t p) {
    #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_ESP8266)
      return (p >= 0) && digitalPinIsValid(p);
    #elif defined(PIN_COUNT)
      return (p >= 0) && (p < (int16_t)PIN_COUNT);
    #elif defined(NUM_DIGITAL_PINS)
      return (p >= 0) && (p < (int16_t)NUM_DIGITAL_PINS);
    #else
      // Best-effort fallback when platform does not expose counts
      return (p >= 0);
    #endif
  }

  int16_t   _pin;
  uint8_t   _polarity;        // HIGH or LOW; pressed when equals _buttonState
  bool      _buttonState;     // debounced level (true == HIGH)
  bool      _lastButtonState; // previous debounced level
  bool      _rawState;        // last raw level (true == HIGH)
  uint32_t  _debounceStart;   // when raw change was last observed
  uint16_t  _debounceMs;      // debounce window in ms
  bool      _hasValidPin;     // true if _pin validated and configured
};