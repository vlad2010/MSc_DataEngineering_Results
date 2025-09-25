#include <stdexcept>
#include <errno.h>

bool Adafruit_HUSB238::getCCStatus() {
    // CWE-476 Fix: Validate pointer before use
    if (i2c_dev == nullptr) {
        // Log error appropriately based on your logging framework
        // For production code, consider using a proper logging mechanism
        throw std::runtime_error("I2C device not initialized");
    }
    
    try {
        // Create register object with proper error handling
        Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
        
        // Validate register creation (assuming the library provides a validation method)
        // If not available in the library, this is a design limitation
        if (!ccRegister.isValid()) {  // Hypothetical validation method
            throw std::runtime_error("Failed to create I2C register");
        }
        
        // Create bit accessor
        Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 6);
        
        // CWE-252 Fix: Check for read errors
        // Assuming the library provides error checking capability
        // If the library doesn't support error checking, consider wrapping with timeout
        bool readSuccess = false;
        uint8_t retryCount = 3;  // Configurable retry count
        bool ccStatus = false;
        
        while (retryCount > 0 && !readSuccess) {
            try {
                ccStatus = ccBit.read();
                readSuccess = true;
            } catch (const std::exception& e) {
                retryCount--;
                if (retryCount == 0) {
                    // Log the specific error
                    throw std::runtime_error(std::string("I2C read failed after retries: ") + e.what());
                }
                // Optional: Add small delay between retries
                // delay(10);  // Platform-specific delay function
            }
        }
        
        return ccStatus;
        
    } catch (const std::bad_alloc& e) {
        // CWE-703 Fix: Handle memory allocation failures
        throw std::runtime_error("Memory allocation failed in getCCStatus");
    } catch (const std::exception& e) {
        // CWE-391 Fix: Proper exception handling and error propagation
        // In embedded systems, you might want to return a default safe value
        // or use an error code pattern instead of exceptions
        
        // Option 1: Re-throw for caller to handle
        throw;
        
        // Option 2: Return safe default with error logging
        // logError("getCCStatus failed: ", e.what());
        // return false;  // Safe default
    }
}

// Alternative implementation using error codes instead of exceptions
// (more suitable for embedded systems with limited resources)
bool Adafruit_HUSB238::getCCStatus(bool* status, int* errorCode) {
    if (status == nullptr || errorCode == nullptr) {
        return false;
    }
    
    *errorCode = 0;
    *status = false;
    
    // Validate I2C device
    if (i2c_dev == nullptr) {
        *errorCode = -ENODEV;  // Device not initialized
        return false;
    }
    
    // Create register object
    Adafruit_I2CRegister ccRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit ccBit = Adafruit_I2CRegisterBit(&ccRegister, 6);
    
    // Attempt read with retries
    uint8_t retryCount = 3;
    while (retryCount > 0) {
        try {
            *status = ccBit.read();
            return true;  // Success
        } catch (...) {
            retryCount--;
            if (retryCount == 0) {
                *errorCode = -EIO;  // I/O error
                return false;
            }
        }
    }
    
    *errorCode = -ETIMEDOUT;  // Timeout error
    return false;
}