#ifndef ADAFRUIT_DEBOUNCE_H
#define ADAFRUIT_DEBOUNCE_H

#include <Arduino.h>
#include <stdint.h>
#include <limits.h>

// Define maximum valid pin number (adjust based on target platform)
#ifndef MAX_PIN_NUMBER
#define MAX_PIN_NUMBER 255
#endif

// Default debounce delay in milliseconds
#ifndef DEFAULT_DEBOUNCE_DELAY
#define DEFAULT_DEBOUNCE_DELAY 50
#endif

class Adafruit_Debounce {
private:
    int16_t _pin;
    bool _polarity;
    bool _buttonState;
    bool _lastButtonState;
    bool _lastDebounceState;
    unsigned long _lastDebounceTime;
    unsigned long _debounceDelay;
    
    // Mutex for thread safety (platform-specific implementation needed)
    // For Arduino without RTOS, we'll use interrupt guards
    
    /**
     * @brief Validate pin number
     * @return true if pin is valid, false otherwise
     */
    bool isValidPin() const {
        return (_pin >= 0 && _pin <= MAX_PIN_NUMBER);
    }
    
    /**
     * @brief Read pin state with validation
     * @return Pin state or last known state if invalid
     */
    bool readPinSafe() {
        if (isValidPin()) {
            return digitalRead(_pin);
        }
        return _buttonState;
    }

public:
    /**
     * @brief Construct a new Adafruit_Debounce object with input validation
     * 
     * @param pin The pin number associated with the button
     * @param polarity The polarity to determine the pressed state (default is LOW)
     * @param debounceDelay The debounce delay in milliseconds (default is 50ms)
     */
    Adafruit_Debounce(int16_t pin, bool polarity = LOW, unsigned long debounceDelay = DEFAULT_DEBOUNCE_DELAY) {
        // Validate and constrain pin number
        if (pin < -1) {
            _pin = -1;  // Invalid pin
        } else if (pin > MAX_PIN_NUMBER) {
            _pin = -1;  // Invalid pin
        } else {
            _pin = pin;
        }
        
        _polarity = polarity;
        _buttonState = !_polarity;
        _lastButtonState = !_polarity;
        _lastDebounceState = !_polarity;
        _lastDebounceTime = 0;
        
        // Validate debounce delay (prevent overflow)
        if (debounceDelay > 10000) {  // Max 10 seconds
            _debounceDelay = 10000;
        } else {
            _debounceDelay = debounceDelay;
        }
    }

    /**
     * @brief Initialize the debounce object with proper validation
     */
    void begin() {
        if (isValidPin()) {
            // Disable interrupts during initialization for thread safety
            noInterrupts();
            
            if (_polarity == LOW) {
                pinMode(_pin, INPUT_PULLUP);
            } else {
                #ifdef INPUT_PULLDOWN
                pinMode(_pin, INPUT_PULLDOWN);
                #else
                pinMode(_pin, INPUT);
                #endif
            }
            
            _buttonState = digitalRead(_pin);
            _lastButtonState = _buttonState;
            _lastDebounceState = _buttonState;
            _lastDebounceTime = millis();
            
            interrupts();
        }
    }

    /**
     * @brief Initialize with a given initial state
     * 
     * @param initialState The initial state to set
     */
    void begin(bool initialState) {
        begin();
        
        // Atomic state update
        noInterrupts();
        _buttonState = initialState;
        _lastButtonState = initialState;
        _lastDebounceState = initialState;
        interrupts();
    }

    /**
     * @brief Read the current debounced state of the button
     * 
     * @return Current debounced button state
     */
    bool read() {
        return _buttonState;
    }

    /**
     * @brief Update the button states with proper debouncing
     * 
     * @param bit The state to set if pin is invalid
     */
    void update(bool bit = false) {
        // Protect state changes
        noInterrupts();
        
        _lastButtonState = _buttonState;
        
        bool currentReading;
        if (isValidPin()) {
            currentReading = digitalRead(_pin);
        } else {
            currentReading = bit;
        }
        
        // Implement actual debouncing logic
        if (currentReading != _lastDebounceState) {
            _lastDebounceTime = millis();
            _lastDebounceState = currentReading;
        }
        
        // Check if debounce delay has passed
        if ((millis() - _lastDebounceTime) > _debounceDelay) {
            // Only update if the state has actually changed
            if (currentReading != _buttonState) {
                _buttonState = currentReading;
            }
        }
        
        interrupts();
    }

    /**
     * @brief Check if the button is currently pressed
     * 
     * @return true if pressed, false otherwise
     */
    bool isPressed() {
        bool result;
        noInterrupts();
        result = (_buttonState == _polarity);
        interrupts();
        return result;
    }

    /**
     * @brief Check if the button is currently released
     * 
     * @return true if released, false otherwise
     */
    bool isReleased() {
        bool result;
        noInterrupts();
        result = (_buttonState != _polarity);
        interrupts();
        return result;
    }

    /**
     * @brief Check if the button was just pressed
     * 
     * @return true if just pressed, false otherwise
     */
    bool justPressed() {
        bool result;
        noInterrupts();
        result = (_buttonState != _lastButtonState) && (_buttonState == _polarity);
        interrupts();
        return result;
    }

    /**
     * @brief Check if the button was just released
     * 
     * @return true if just released, false otherwise
     */
    bool justReleased() {
        bool result;
        noInterrupts();
        result = (_buttonState != _lastButtonState) && (_buttonState != _polarity);
        interrupts();
        return result;
    }
    
    /**
     * @brief Set debounce delay
     * 
     * @param delay Delay in milliseconds (capped at 10000ms)
     */
    void setDebounceDelay(unsigned long delay) {
        if (delay > 10000) {
            _debounceDelay = 10000;
        } else {
            _debounceDelay = delay;
        }
    }
};

#endif // ADAFRUIT_DEBOUNCE_H