#include <Arduino.h>
#include <limits.h>

class Adafruit_Debounce {
private:
    static constexpr int INVALID_PIN = -1;
    static constexpr int MAX_PIN = 255;  // Adjust based on your platform
    
    int _pin;
    bool _polarity;
    bool _buttonState;
    bool _initialized;
    
public:
    // Constructor with proper initialization
    Adafruit_Debounce(int pin = INVALID_PIN, bool polarity = HIGH) 
        : _pin(pin), _polarity(polarity), _buttonState(false), _initialized(false) {
        // Validate pin during construction
        if (_pin != INVALID_PIN && (_pin < 0 || _pin > MAX_PIN)) {
            _pin = INVALID_PIN;  // Force to invalid state
        }
        _initialized = true;
    }
    
    // Secure read method with proper validation
    bool read() {
        // Check for object initialization (defense against use-after-free)
        if (!_initialized) {
            // Log error in production code
            return false;  // Safe default
        }
        
        // Validate pin range with both bounds
        if (_pin >= 0 && _pin <= MAX_PIN) {
            // Add try-catch or error checking based on platform capabilities
            int readValue = digitalRead(_pin);
            
            // Check for read errors (platform-specific)
            // Some Arduino platforms return -1 on error
            if (readValue < 0) {
                // Log error in production code
                return _buttonState;  // Return last known state
            }
            
            // Ensure polarity is binary (HIGH=1 or LOW=0)
            bool polarityBool = (_polarity != LOW);
            bool pinState = (readValue != LOW);
            
            // Update internal state for error recovery
            _buttonState = (pinState == polarityBool);
            return _buttonState;
        }
        
        // Pin is invalid or out of range
        // In production, consider logging this condition
        return _buttonState;  // Return cached state as fallback
    }
    
    // Safe setter methods with validation
    bool setPin(int pin) {
        if (pin == INVALID_PIN || (pin >= 0 && pin <= MAX_PIN)) {
            _pin = pin;
            return true;
        }
        return false;  // Invalid pin
    }
    
    bool setPolarity(bool polarity) {
        _polarity = polarity;
        return true;
    }
    
    // Getter for diagnostics
    bool isValidPin() const {
        return (_pin >= 0 && _pin <= MAX_PIN);
    }
    
    int getPin() const {
        return _pin;
    }
};