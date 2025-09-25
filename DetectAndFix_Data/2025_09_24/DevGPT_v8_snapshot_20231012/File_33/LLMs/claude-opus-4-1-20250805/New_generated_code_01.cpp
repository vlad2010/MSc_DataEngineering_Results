#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include <Arduino.h>
#include <stdint.h>
#include <limits.h>

class Adafruit_Debounce {
private:
    static constexpr int16_t INVALID_PIN = -1;
    static constexpr int16_t MIN_VALID_PIN = 0;
    static constexpr int16_t MAX_VALID_PIN = 255;  // Adjust based on platform
    static constexpr uint32_t DEFAULT_DEBOUNCE_DELAY = 50; // milliseconds
    static constexpr uint32_t MAX_DEBOUNCE_DELAY = 1000;   // 1 second max
    
    int16_t _pin;
    bool _polarity;
    bool _buttonState;
    bool _lastButtonState;
    bool _currentState;
    bool _previousState;
    uint32_t _lastDebounceTime;
    uint32_t _debounceDelay;
    bool _justPressedFlag;
    bool _justReleasedFlag;
    bool _initialized;
    
    // Validate pin number
    bool isValidPin(int16_t pin) const {
        return (pin >= MIN_VALID_PIN && pin <= MAX_VALID_PIN);
    }
    
    // Safe time difference calculation to prevent overflow
    uint32_t safeTimeDiff(uint32_t current, uint32_t previous) const {
        if (current >= previous) {
            return current - previous;
        }
        // Handle overflow case
        return (UINT32_MAX - previous) + current + 1;
    }

public:
    Adafruit_Debounce(int16_t pin, bool polarity = LOW, uint32_t debounceDelay = DEFAULT_DEBOUNCE_DELAY) 
        : _pin(INVALID_PIN),
          _polarity(polarity),
          _buttonState(!polarity),
          _lastButtonState(!polarity),
          _currentState(!polarity),
          _previousState(!polarity),
          _lastDebounceTime(0),
          _debounceDelay(DEFAULT_DEBOUNCE_DELAY),
          _justPressedFlag(false),
          _justReleasedFlag(false),
          _initialized(false) {
        
        // Input validation for pin
        if (isValidPin(pin)) {
            _pin = pin;
        }
        
        // Input validation for debounce delay
        if (debounceDelay > 0 && debounceDelay <= MAX_DEBOUNCE_DELAY) {
            _debounceDelay = debounceDelay;
        }
    }
    
    bool begin() {
        // Validate pin before initialization
        if (!isValidPin(_pin)) {
            return false;
        }
        
        // Set pin mode with pull-up resistor for better noise immunity
        if (_polarity == LOW) {
            pinMode(_pin, INPUT_PULLUP);
        } else {
            pinMode(_pin, INPUT);
        }
        
        // Initialize the current state
        _currentState = digitalRead(_pin);
        _buttonState = _currentState;
        _lastButtonState = _currentState;
        _previousState = _currentState;
        _lastDebounceTime = millis();
        _initialized = true;
        
        return true;
    }
    
    bool read() {
        if (!_initialized || !isValidPin(_pin)) {
            return false;
        }
        
        // Protect against interrupt-based modifications
        noInterrupts();
        
        bool reading = digitalRead(_pin);
        uint32_t currentTime = millis();
        
        // Reset flags
        _justPressedFlag = false;
        _justReleasedFlag = false;
        
        // Check if the button state has changed
        if (reading != _lastButtonState) {
            _lastDebounceTime = currentTime;
        }
        
        // Check if enough time has passed since the last state change
        if (safeTimeDiff(currentTime, _lastDebounceTime) > _debounceDelay) {
            // Update the button state only if it has stabilized
            if (reading != _buttonState) {
                _previousState = _buttonState;
                _buttonState = reading;
                
                // Detect state transitions
                if (_buttonState == _polarity) {
                    _justPressedFlag = true;
                } else {
                    _justReleasedFlag = true;
                }
            }
        }
        
        _lastButtonState = reading;
        _currentState = _buttonState;
        
        interrupts();
        
        return (_currentState == _polarity);
    }
    
    void update(bool bit) {
        if (!_initialized) {
            return;
        }
        
        // Protect against concurrent access
        noInterrupts();
        
        uint32_t currentTime = millis();
        
        _justPressedFlag = false;
        _justReleasedFlag = false;
        
        if (bit != _lastButtonState) {
            _lastDebounceTime = currentTime;
        }
        
        if (safeTimeDiff(currentTime, _lastDebounceTime) > _debounceDelay) {
            if (bit != _buttonState) {
                _previousState = _buttonState;
                _buttonState = bit;
                
                if (_buttonState == _polarity) {
                    _justPressedFlag = true;
                } else {
                    _justReleasedFlag = true;
                }
            }
        }
        
        _lastButtonState = bit;
        _currentState = _buttonState;
        
        interrupts();
    }
    
    bool isPressed() const {
        if (!_initialized) {
            return false;
        }
        return (_currentState == _polarity);
    }
    
    bool isReleased() const {
        if (!_initialized) {
            return false;
        }
        return (_currentState != _polarity);
    }
    
    bool justPressed() const {
        if (!_initialized) {
            return false;
        }
        return _justPressedFlag;
    }
    
    bool justReleased() const {
        if (!_initialized) {
            return false;
        }
        return _justReleasedFlag;
    }
    
    // Additional safety methods
    bool isInitialized() const {
        return _initialized;
    }
    
    void setDebounceDelay(uint32_t delay) {
        if (delay > 0 && delay <= MAX_DEBOUNCE_DELAY) {
            noInterrupts();
            _debounceDelay = delay;
            interrupts();
        }
    }
    
    uint32_t getDebounceDelay() const {
        return _debounceDelay;
    }
};

#endif // ADAFRUIT_DEBOUNCE_H