// Adafruit_Debounce.h/.cpp (single-file example)
// Hardened implementation with input validation, deterministic input conditioning,
// rollover-safe debounce, and concurrency-safe state transitions for Arduino-like cores.

#include <Arduino.h>

// Critical section helpers: save/restore interrupt state on AVR; best-effort elsewhere.
#if defined(__AVR__)
  #include <avr/interrupt.h>
  #define CRITICAL_SECTION_START uint8_t _sreg = SREG; cli();
  #define CRITICAL_SECTION_END   SREG = _sreg;
#else
  // Note: On non-AVR platforms, we can't easily restore prior interrupt mask portably.
  // This is a best-effort guard. Avoid calling these methods from within an ISR if possible.
  #define CRITICAL_SECTION_START noInterrupts();
  #define CRITICAL_SECTION_END   interrupts();
#endif

// Best-effort pin validation across cores
static inline bool isPinValid(int16_t p) {
  #if defined(NUM_DIGITAL_PINS)
    return (p >= 0) && (p < (int16_t)NUM_DIGITAL_PINS);
  #elif defined(PIN_COUNT)
    return (p >= 0) && (p < (int16_t)PIN_COUNT);
  #else
    // Fallback: cannot validate upper bound; at least reject negative
    return (p >= 0);
  #endif
}

class Adafruit_Debounce {
public:
  // activeHigh: true if the "pressed" electrical level is HIGH, false if LOW (active-low)
  // debounceMs: debounce interval in milliseconds (default 30ms; clamped to sane bounds)
  Adafruit_Debounce(int16_t pin, bool activeHigh, unsigned long debounceMs = 30)
  : _pin(pin),
    _activeHigh(activeHigh),
    _validPin(false),
    _debounceDelay(clampDebounce(debounceMs)),
    _lastChangeTime(0),
    _rawState(0),
    _lastRawState(0),
    _stableState(0),
    _justPressedFlag(0),
    _justReleasedFlag(0)
  {}

  // Backwards-compatible begin(): assume internal pull-up for active-low buttons,
  // plain INPUT for active-high. You can override with begin(true/false).
  void begin() {
    bool usePullup = !_activeHigh; // typical wiring: active-low with pull-up
    begin(usePullup);
  }

  // Explicit begin with option for internal pull-up resistor
  void begin(bool internalPullup) {
    _validPin = isPinValid(_pin);
    if (!_validPin) {
      // Fail closed: remain "not pressed", no transitions ever emitted.
      return;
    }

    if (internalPullup) {
      pinMode(_pin, INPUT_PULLUP);
    } else {
      pinMode(_pin, INPUT);
    }

    // Read actual hardware level to avoid spurious first-event (CWE-665 fix).
    uint8_t raw = (uint8_t)digitalRead(_pin);
    CRITICAL_SECTION_START
      _rawState = raw;
      _lastRawState = raw;
      _stableState = raw;
      _lastChangeTime = millis();
      _justPressedFlag = 0;
      _justReleasedFlag = 0;
    CRITICAL_SECTION_END
  }

  // Poll the pin, run debounce state machine, and return current pressed state.
  // Returns false if pin invalid.
  bool read() {
    if (!_validPin) return false;

    uint8_t raw = (uint8_t)digitalRead(_pin);
    return processRaw(raw);
  }

  // Manually feed a raw level (0/LOW or 1/HIGH), e.g., from a different sampling context.
  // Returns the current pressed state after debouncing.
  bool update(bool rawLevel) {
    if (!_validPin) return false;
    uint8_t raw = rawLevel ? HIGH : LOW;
    return processRaw(raw);
  }

  // Debounced "pressed" state
  bool isPressed() {
    if (!_validPin) return false;
    uint8_t stable;
    CRITICAL_SECTION_START
      stable = _stableState;
    CRITICAL_SECTION_END
    return stable == (uint8_t)(_activeHigh ? HIGH : LOW);
  }

  bool isReleased() {
    return !isPressed();
  }

  // One-shot event flags. Reading clears the flag atomically.
  bool justPressed() {
    if (!_validPin) return false;
    uint8_t flag;
    CRITICAL_SECTION_START
      flag = _justPressedFlag;
      _justPressedFlag = 0;
    CRITICAL_SECTION_END
    return flag != 0;
  }

  bool justReleased() {
    if (!_validPin) return false;
    uint8_t flag;
    CRITICAL_SECTION_START
      flag = _justReleasedFlag;
      _justReleasedFlag = 0;
    CRITICAL_SECTION_END
    return flag != 0;
  }

  // Optionally adjust debounce delay; clamped to [1, 1000] ms to avoid pathological values.
  void setDebounce(unsigned long ms) {
    CRITICAL_SECTION_START
      _debounceDelay = clampDebounce(ms);
    CRITICAL_SECTION_END
  }

  bool valid() const { return _validPin; }

private:
  static unsigned long clampDebounce(unsigned long ms) {
    if (ms < 1UL)   return 1UL;
    if (ms > 1000UL) return 1000UL; // cap to 1s to avoid DOS-like settings
    return ms;
  }

  // Core debounce logic with rollover-safe timing (CWE-190 fix).
  bool processRaw(uint8_t raw) {
    unsigned long now = millis();

    // Edge on raw input: reset debounce timer
    // No critical section needed for simple byte write on most MCUs, but safe anyway.
    if (raw != _lastRawState) {
      CRITICAL_SECTION_START
        _lastChangeTime = now;
        _lastRawState = raw;
      CRITICAL_SECTION_END
    }

    // Rollover-safe: difference in unsigned long
    bool stateChanged = false;
    CRITICAL_SECTION_START
      if ((unsigned long)(now - _lastChangeTime) >= _debounceDelay) {
        if (raw != _stableState) {
          // Stable transition after debounce interval
          uint8_t prevStable = _stableState;
          _stableState = raw;
          stateChanged = true;

          // Compute "pressed" transitions relative to active level
          bool wasPressed = prevStable == (uint8_t)(_activeHigh ? HIGH : LOW);
          bool isNowPressed = _stableState == (uint8_t)(_activeHigh ? HIGH : LOW);

          if (!wasPressed && isNowPressed) {
            _justPressedFlag = 1;
          } else if (wasPressed && !isNowPressed) {
            _justReleasedFlag = 1;
          }
        }
      }
    CRITICAL_SECTION_END

    // Return current pressed state
    return (_stableState == (uint8_t)(_activeHigh ? HIGH : LOW));
  }

  // Configuration
  int16_t _pin;
  bool _activeHigh;
  bool _validPin;

  // Debounce timing
  volatile unsigned long _lastChangeTime;
  volatile unsigned long _debounceDelay;

  // States and event flags (volatile for ISR/main concurrency safety)
  volatile uint8_t _rawState;
  volatile uint8_t _lastRawState;
  volatile uint8_t _stableState;
  volatile uint8_t _justPressedFlag;
  volatile uint8_t _justReleasedFlag;
};