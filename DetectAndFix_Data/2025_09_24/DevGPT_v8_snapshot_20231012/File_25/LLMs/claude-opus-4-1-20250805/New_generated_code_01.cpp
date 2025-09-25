#include <stdexcept>
#include <limits>

// Assuming enum definition (adjust based on actual implementation)
enum HUSB238_5VCurrentContract {
    CONTRACT_DEFAULT = 0,
    CONTRACT_1_5A = 1,
    CONTRACT_2_4A = 2,
    CONTRACT_3A = 3,
    CONTRACT_INVALID = -1  // Add invalid state for error handling
};

class Adafruit_HUSB238 {
private:
    Adafruit_I2CDevice* i2c_dev;
    static constexpr uint8_t HUSB238_PD_STATUS0 = 0x00; // Adjust address as needed
    static constexpr uint8_t MAX_CONTRACT_VALUE = 3;    // 2 bits = max value 3
    static constexpr uint8_t CONTRACT_BIT_MASK = 0x03; // Mask for 2 bits
    
public:
    HUSB238_5VCurrentContract get5VContractA() {
        // Validate i2c_dev pointer
        if (i2c_dev == nullptr) {
            // Log error if logging is available
            return CONTRACT_INVALID;
        }
        
        try {
            // Create an Adafruit_I2CRegister object for the PD_STATUS0 register
            Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0);
            
            // Create an Adafruit_I2CRegisterBits object for the bottom two bits (0-1) of PD_STATUS0
            Adafruit_I2CRegisterBits contractABits = Adafruit_I2CRegisterBits(&pdRegister, 2, 0);
            
            // Read the bits with error checking
            uint8_t rawValue = contractABits.read();
            
            // Apply mask to ensure we only use the relevant bits
            rawValue &= CONTRACT_BIT_MASK;
            
            // Validate the read value is within expected range
            if (rawValue > MAX_CONTRACT_VALUE) {
                // This shouldn't happen with 2-bit mask, but defensive programming
                return CONTRACT_INVALID;
            }
            
            // Validate against known enum values
            switch(rawValue) {
                case CONTRACT_DEFAULT:
                case CONTRACT_1_5A:
                case CONTRACT_2_4A:
                case CONTRACT_3A:
                    return static_cast<HUSB238_5VCurrentContract>(rawValue);
                default:
                    // Unexpected value - should not reach here with proper masking
                    return CONTRACT_INVALID;
            }
            
        } catch (const std::exception& e) {
            // Handle I2C communication errors
            // Log the error if logging is available
            // Serial.print("I2C Error in get5VContractA: ");
            // Serial.println(e.what());
            return CONTRACT_INVALID;
        } catch (...) {
            // Catch any other unexpected errors
            return CONTRACT_INVALID;
        }
    }
    
    // Helper method to check if the contract reading was successful
    bool isValidContract(HUSB238_5VCurrentContract contract) {
        return (contract >= CONTRACT_DEFAULT && contract <= CONTRACT_3A);
    }
};