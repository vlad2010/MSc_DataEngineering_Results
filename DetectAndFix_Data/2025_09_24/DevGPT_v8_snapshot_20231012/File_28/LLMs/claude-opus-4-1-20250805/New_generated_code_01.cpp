#include <stdexcept>
#include <limits>

// Assuming enum definition (adjust based on actual implementation)
enum HUSB238_ResponseCodes {
    HUSB238_RESPONSE_OK = 0,
    HUSB238_RESPONSE_ERROR = 1,
    // ... other valid codes
    HUSB238_RESPONSE_MAX = 7  // Maximum valid value for 3 bits
};

class Adafruit_HUSB238 {
private:
    Adafruit_I2CDevice* i2c_dev;
    static constexpr uint8_t HUSB238_PD_STATUS1 = 0x00; // Adjust to actual register address
    
    // Helper function to validate enum value
    bool isValidResponseCode(uint8_t value) const {
        return value <= HUSB238_RESPONSE_MAX;
    }

public:
    HUSB238_ResponseCodes getPDResponse() {
        // Input validation
        if (i2c_dev == nullptr) {
            throw std::runtime_error("I2C device not initialized");
        }
        
        try {
            // Create register with error handling
            Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
            
            // Verify register was created successfully (implementation-specific check)
            if (!pdRegister.isValid()) {  // Assuming such method exists
                throw std::runtime_error("Failed to create I2C register");
            }
            
            // Create bit accessor - assuming the parameters are (register, start_bit, num_bits)
            // If it's actually (register, start_bit, end_bit), then it should be (3, 5) for bits 3-5
            Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);
            
            // Read value with bounds checking
            uint8_t rawValue = pdBits.read();
            
            // Validate the read value is within expected range (3 bits = 0-7)
            if (rawValue > 0b111) {
                throw std::out_of_range("Read value exceeds 3-bit range");
            }
            
            // Validate against enum range
            if (!isValidResponseCode(rawValue)) {
                // Log the unexpected value for debugging
                // logger.warning("Unexpected PD response code: %d", rawValue);
                
                // Return a safe default or error code
                return HUSB238_RESPONSE_ERROR;
            }
            
            // Safe cast after validation
            return static_cast<HUSB238_ResponseCodes>(rawValue);
            
        } catch (const std::exception& e) {
            // Log the error (implementation-specific)
            // logger.error("Failed to read PD response: %s", e.what());
            
            // Return error code or rethrow based on requirements
            return HUSB238_RESPONSE_ERROR;
        }
    }
    
    // Alternative implementation with error code output parameter
    HUSB238_ResponseCodes getPDResponseSafe(bool* success) {
        if (success != nullptr) {
            *success = false;
        }
        
        if (i2c_dev == nullptr) {
            return HUSB238_RESPONSE_ERROR;
        }
        
        try {
            Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
            Adafruit_I2CRegisterBits pdBits = Adafruit_I2CRegisterBits(&pdRegister, 3, 3);
            
            uint8_t rawValue = pdBits.read();
            
            if (rawValue > 0b111 || !isValidResponseCode(rawValue)) {
                return HUSB238_RESPONSE_ERROR;
            }
            
            if (success != nullptr) {
                *success = true;
            }
            
            return static_cast<HUSB238_ResponseCodes>(rawValue);
            
        } catch (...) {
            return HUSB238_RESPONSE_ERROR;
        }
    }
};