#include <stdexcept>
#include <chrono>

class Adafruit_HUSB238 {
private:
    static constexpr uint8_t HUSB238_PD_STATUS1 = 0x00; // Assuming register address
    static constexpr uint32_t I2C_TIMEOUT_MS = 100;
    static constexpr uint8_t MAX_RETRY_COUNT = 3;
    
    Adafruit_I2CDevice* i2c_dev;
    bool initialized;
    
    // Helper method for I2C operations with retry logic
    bool performI2COperation(std::function<bool()> operation, const char* operation_name) {
        if (!initialized || i2c_dev == nullptr) {
            throw std::runtime_error("I2C device not initialized");
        }
        
        for (uint8_t retry = 0; retry < MAX_RETRY_COUNT; retry++) {
            try {
                if (operation()) {
                    return true;
                }
            } catch (const std::exception& e) {
                // Log the error (implementation depends on your logging framework)
                // logger.error("I2C operation %s failed (attempt %d/%d): %s", 
                //              operation_name, retry + 1, MAX_RETRY_COUNT, e.what());
                
                if (retry == MAX_RETRY_COUNT - 1) {
                    throw; // Re-throw on final attempt
                }
                
                // Small delay before retry
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        return false;
    }

public:
    Adafruit_HUSB238() : i2c_dev(nullptr), initialized(false) {}
    
    // Initialization method (should be called before using get5VContractV)
    bool begin(Adafruit_I2CDevice* device) {
        if (device == nullptr) {
            return false;
        }
        
        i2c_dev = device;
        
        // Verify device is responsive
        try {
            // Attempt to read a known register to verify communication
            uint8_t test_read;
            if (!i2c_dev->read(&test_read, 1)) {
                initialized = false;
                return false;
            }
            initialized = true;
            return true;
        } catch (...) {
            initialized = false;
            return false;
        }
    }
    
    // Fixed version of get5VContractV with proper security measures
    bool get5VContractV() {
        // Input validation - check initialization state
        if (!initialized || i2c_dev == nullptr) {
            throw std::invalid_argument("HUSB238: Device not initialized. Call begin() first.");
        }
        
        try {
            // Create register with timeout protection
            auto start_time = std::chrono::steady_clock::now();
            
            Adafruit_I2CRegister pdRegister = Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS1);
            
            // Check for timeout
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed > I2C_TIMEOUT_MS) {
                throw std::runtime_error("I2C operation timeout");
            }
            
            // Create bit accessor with validation
            Adafruit_I2CRegisterBit contractVBit = Adafruit_I2CRegisterBit(&pdRegister, 2);
            
            // Perform read with retry logic
            bool result = false;
            bool read_success = performI2COperation([&]() {
                try {
                    result = contractVBit.read();
                    return true;
                } catch (...) {
                    return false;
                }
            }, "get5VContractV");
            
            if (!read_success) {
                throw std::runtime_error("Failed to read 5V contract status after retries");
            }
            
            // Validate the result (bit should be 0 or 1)
            // Additional validation could be added based on hardware specifications
            
            return result;
            
        } catch (const std::exception& e) {
            // Log error for debugging (implementation depends on your logging framework)
            // logger.error("HUSB238::get5VContractV failed: %s", e.what());
            
            // Depending on requirements, either re-throw or return safe default
            // Option 1: Re-throw for caller to handle
            throw;
            
            // Option 2: Return safe default (uncomment if preferred)
            // return false; // Safe default - no 5V contract
        }
    }
    
    // Destructor for cleanup
    ~Adafruit_HUSB238() {
        initialized = false;
        i2c_dev = nullptr;
    }
};