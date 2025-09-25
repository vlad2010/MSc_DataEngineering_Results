#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include <Arduino.h>

// Simple RAII guard for short critical sections where we touch multi-byte shared state.
// Keep critical sections as short as possible.
class InterruptGuard {
public:
  InterruptGuard() { noInterrupts(); }
  ~InterruptGuard() { interrupts(); }
};

class Adafruit_Debounce {
public:
  // polarity: LOW => button is "pressed" when pin reads LOW (typical with INPUT_PULLUP)
  //           HIGH => button is "pressed" when pin reads HIGH (external pull-down)
  explicit Adafruit_Debounce(int16_t pin, bool polarity = LOW, uint16_t debounceMs = 20)
    : _pin(pin),
      _polarity(polarity),
      _debounceMs(debounceMs),
      _lastDebounceTime(0),
      _buttonState(false),
      _lastButtonState(false),
      _lastRaw(false),
      _justPressed(false),
      _justReleased(false) {}

  // Must be called before use (from setup). Initializes pin mode and state.
  void begin() {
    if (!validPin()) {
      return; // avoid invalid pin use
    }

    // Choose pin mode that matches active level. Active-low typically uses pull-up.
    if (_polarity == LOW) {
      pinMode(_pin, INPUT_PULLUP);
    } else {
      pinMode(_pin, INPUT);
    }

    // Initialize state from current hardware level
    bool rawHigh = (digitalRead(_pin) == HIGH);
    bool pressedRaw = (rawHigh == (_polarity == HIGH));
    uint32_t now = millis();

    InterruptGuard guard; // atomic update of shared state
    _lastDebounceTime = now;
    _buttonState = pressedRaw;
    _lastButtonState = pressedRaw;
    _lastRaw = pressedRaw;
    _justPressed = false;
    _justReleased = false;
  }

  // Poll the hardware and return the debounced logical "pressed" state.
  // Safe for main loop; do not call from ISR due to millis().
  bool read() {
    if (!validPin()) {
      return false;
    }
    bool rawHigh = (digitalRead(_pin) == HIGH);
    update(rawHigh); // debounces and updates internal state
    return isPressed();
  }

  // Provide latest raw level that was read elsewhere (true => HIGH).
  // Safe for main loop; do not call from ISR due to millis().
  void update(bool rawHigh) {
    bool pressedRaw = (rawHigh == (_polarity == HIGH));
    uint32_t now = millis();

    // Critical section for multi-byte shared state on 8-bit MCUs
    InterruptGuard guard;

    // If raw state changed, reset debounce timer
    if (pressedRaw != _lastRaw) {
      _lastDebounceTime = now;
      _lastRaw = pressedRaw;
    }

    // Apply debounce after interval elapsed (rollover-safe)
    if ((uint32_t)(now - _lastDebounceTime) >= _debounceMs) {
      if (pressedRaw != _buttonState) {
        _lastButtonState = _buttonState;
        _buttonState = pressedRaw;
        _justPressed = (!_lastButtonState && _buttonState);
        _justReleased = (_lastButtonState && !_buttonState);
      } else {
        // No edge this cycle
        _justPressed = false;
        _justReleased = false;
      }
    }
  }

  bool isPressed() const {
    InterruptGuard guard;
    return _buttonState;
  }

  bool isReleased() const {
    InterruptGuard guard;
    return !_buttonState;
  }

  // One-shot edge latches; reading clears the latch.
  bool justPressed() {
    InterruptGuard guard;
    bool jp = _justPressed;
    _justPressed = false;
    return jp;
  }

  bool justReleased() {
    InterruptGuard guard;
    bool jr = _justReleased;
    _justReleased = false;
    return jr;
  }

private:
  bool validPin() const {
  #ifdef NUM_DIGITAL_PINS
    return (_pin >= 0) && (_pin < NUM_DIGITAL_PINS);
  #else
    return (_pin >= 0);
  #endif
  }

  int16_t _pin;
  bool _polarity;            // LOW = active-low, HIGH = active-high
  uint16_t _debounceMs;      // debounce interval in ms

  // Volatile because these can be observed/updated across contexts;
  // guarded by short critical sections to avoid torn 32-bit accesses on 8-bit MCUs.
  volatile uint32_t _lastDebounceTime;
  volatile bool _buttonState;      // debounced logical pressed state
  volatile bool _lastButtonState;  // previous debounced state (for edge detection)
  volatile bool _lastRaw;          // last raw pressed state
  volatile bool _justPressed;      // edge latch
  volatile bool _justReleased;     // edge latch
};

#endif // ADAFRUIT_DEBOUNCE_H