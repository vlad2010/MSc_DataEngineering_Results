#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include "Arduino.h"
#include <stdint.h>
#include <limits.h>

// Define maximum valid pin number (adjust based on target board)
#ifndef MAX_ARDUINO_PIN
#define MAX_ARDUINO_PIN 255
#endif

// Default debounce delay in milliseconds
#ifndef DEFAULT_DEBOUNCE_DELAY
#define DEFAULT_DEBOUNCE_DELAY 50
#endif

class Adafruit_Debounce {
  public:
    // Constructor with input validation and default initialization
    explicit Adafruit_Debounce(int16_t pin, bool polarity = LOW, uint32_t debounceDelay = DEFAULT_DEBOUNCE_DELAY) 
        : _pin(pin), 
          _polarity(polarity),
          _debounceDelay(debounceDelay),
          _lastDebounceTime(0),
          _buttonState(false),
          _lastButtonState(false),
          _initialized(false),
          _error(false) {
        // Validate pin number
        if (pin < 0 || pin > MAX_ARDUINO_PIN) {
            _error = true;
            _pin = -1; // Invalid pin marker
        }
        // Validate debounce delay to prevent overflow
        if (debounceDelay > (UINT32_MAX / 2)) {
            _debounceDelay = DEFAULT_DEBOUNCE_DELAY;
        }
    }
    
    // Initialize the pin with error checking
    bool begin() {
        if (_error || _pin < 0) {
            return false;
        }
        
        // Set pin mode with internal pull-up if needed
        if (_polarity == LOW) {
            pinMode(_pin, INPUT_PULLUP);
        } else {
            pinMode(_pin, INPUT);
        }
        
        // Initialize states
        _buttonState = digitalRead(_pin) == _polarity;
        _lastButtonState = _buttonState;
        _lastDebounceTime = millis();
        _initialized = true;
        
        return true;
    }
    
    // Read current debounced state with error checking
    bool read() const {
        if (!_initialized || _error) {
            return false;
        }
        return _buttonState;
    }
    
    // Update with external bit value (bounds checking for time)
    void update(bool bit) {
        if (!_initialized || _error) {
            return;
        }
        
        uint32_t currentTime = millis();
        
        // Handle timer overflow
        if (currentTime < _lastDebounceTime) {
            _lastDebounceTime = currentTime;
        }
        
        // Check if enough time has passed since last change
        if ((currentTime - _lastDebounceTime) >= _debounceDelay) {
            if (bit != _lastButtonState) {
                _lastDebounceTime = currentTime;
                _lastButtonState = bit;
            }
            _buttonState = bit;
        }
    }
    
    // Update by reading from pin
    bool update() {
        if (!_initialized || _error || _pin < 0) {
            return false;
        }
        
        bool currentReading = (digitalRead(_pin) == _polarity);
        update(currentReading);
        return _buttonState;
    }
    
    // Const-correct query methods
    bool isPressed() const {
        if (!_initialized || _error) {
            return false;
        }
        return _buttonState;
    }
    
    bool isReleased() const {
        if (!_initialized || _error) {
            return true;  // Safe default
        }
        return !_buttonState;
    }
    
    bool justPressed() const {
        if (!_initialized || _error) {
            return false;
        }
        return (_buttonState && !_lastButtonState);
    }
    
    bool justReleased() const {
        if (!_initialized || _error) {
            return false;
        }
        return (!_buttonState && _lastButtonState);
    }
    
    // Error checking methods
    bool isValid() const {
        return _initialized && !_error;
    }
    
    bool hasError() const {
        return _error;
    }
    
    // Get current configuration (const-correct)
    int16_t getPin() const { return _pin; }
    bool getPolarity() const { return _polarity; }
    uint32_t getDebounceDelay() const { return _debounceDelay; }

  private:
    int16_t _pin;                    // Pin number (-1 if invalid)
    bool _polarity;                  // Active HIGH or LOW
    uint32_t _debounceDelay;         // Debounce delay in milliseconds
    uint32_t _lastDebounceTime;      // Last time the pin state changed
    bool _buttonState;               // Current debounced state
    bool _lastButtonState;           // Previous reading for edge detection
    bool _initialized;               // Track initialization status
    bool _error;                     // Error flag for invalid configuration
    
    // Prevent copying (Rule of Three/Five)
    Adafruit_Debounce(const Adafruit_Debounce&) = delete;
    Adafruit_Debounce& operator=(const Adafruit_Debounce&) = delete;
};

#endif // ADAFRUIT_DEBOUNCE_H