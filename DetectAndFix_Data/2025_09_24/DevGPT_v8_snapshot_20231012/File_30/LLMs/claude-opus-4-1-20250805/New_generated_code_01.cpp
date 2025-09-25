#include <stdexcept>
#include <limits>

// Assuming enum definition (adjust based on actual implementation)
enum HUSB238_5VCurrentContract {
    HUSB238_5V_CONTRACT_DEFAULT = 0,
    HUSB238_5V_CONTRACT_1_5A = 1,
    HUSB238_5V_CONTRACT_2_4A = 2,
    HUSB238_5V_CONTRACT_3A = 3,
    HUSB238_5V_CONTRACT_MAX = 3  // Maximum valid value
};

class Adafruit_HUSB238 {
private:
    // Validate enum value is within valid range
    bool isValid5VContract(uint8_t value) const {
        return value <= HUSB238_5V_CONTRACT_MAX;
    }
    
public:
    HUSB238_5VCurrentContract get5VContractA() {
        // Input validation
        if (!i2c_dev) {
            throw std::runtime_error("I2C device not initialized");
        }
        
        try {
            // Create register with error checking
            Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
            
            // Verify register was created successfully (implementation-specific check)
            // This assumes the library provides some way to check validity
            if (!pdRegister.isValid()) {  // Adjust based on actual API
                throw std::runtime_error("Failed to create I2C register");
            }
            
            // Create register bits accessor
            Adafruit_I2CRegisterBits contractABits = 
                Adafruit_I2CRegisterBits(&pdRegister, 2, 0);  // Bottom two bits (0-1)
            
            // Read the value with error checking
            uint8_t rawValue = contractABits.read();
            
            // Validate the read value is within expected range
            // 2 bits can only represent values 0-3
            if (rawValue > 3) {
                throw std::runtime_error("Invalid contract value read from register: " + 
                                       std::to_string(rawValue));
            }
            
            // Additional validation against enum range
            if (!isValid5VContract(rawValue)) {
                // Log warning or handle gracefully
                // Could return a default safe value or throw exception
                throw std::out_of_range("Contract value out of valid enum range: " + 
                                       std::to_string(rawValue));
            }
            
            // Safe cast after validation
            return static_cast<HUSB238_5VCurrentContract>(rawValue);
            
        } catch (const std::exception& e) {
            // Log the error (implementation-specific logging)
            // logger.error("Failed to read 5V contract: " + std::string(e.what()));
            
            // Depending on requirements, either:
            // Option 1: Re-throw to let caller handle
            throw;
            
            // Option 2: Return safe default value
            // return HUSB238_5V_CONTRACT_DEFAULT;
        }
    }
    
    // Alternative implementation with error code pattern
    bool get5VContractA_safe(HUSB238_5VCurrentContract& contract) {
        try {
            contract = get5VContractA();
            return true;
        } catch (...) {
            contract = HUSB238_5V_CONTRACT_DEFAULT;
            return false;
        }
    }
};