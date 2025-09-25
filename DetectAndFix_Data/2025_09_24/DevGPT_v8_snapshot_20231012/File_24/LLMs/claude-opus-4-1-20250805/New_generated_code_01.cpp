#include "Adafruit_HUSB238.h"
#include <Adafruit_BusIO.h>
#include <Adafruit_I2CRegister.h>
#include <Wire.h>

// HUSB238 default I2C address (verify with datasheet)
#define HUSB238_I2CADDR_DEFAULT 0x08  // Common address for HUSB238

// Register definitions (verify with datasheet)
#define HUSB238_PD_STATUS0 0x00  // Example register address

class Adafruit_HUSB238 {
private:
    Adafruit_I2CDevice* i2c_dev;
    uint8_t i2c_address;
    bool initialized;
    
public:
    // Constructor with default I2C address
    Adafruit_HUSB238(uint8_t addr = HUSB238_I2CADDR_DEFAULT) 
        : i2c_dev(nullptr), i2c_address(addr), initialized(false) {
        // Validate I2C address range (7-bit addressing)
        if (addr > 0x7F) {
            i2c_address = HUSB238_I2CADDR_DEFAULT;
        }
    }
    
    // Destructor for proper cleanup
    ~Adafruit_HUSB238() {
        if (i2c_dev != nullptr) {
            delete i2c_dev;
            i2c_dev = nullptr;
        }
        initialized = false;
    }
    
    // Initialize the device
    bool begin(TwoWire* theWire = &Wire) {
        // Prevent double initialization
        if (initialized) {
            return true;
        }
        
        // Clean up any existing device
        if (i2c_dev != nullptr) {
            delete i2c_dev;
        }
        
        // Create new I2C device with error checking
        try {
            i2c_dev = new Adafruit_I2CDevice(i2c_address, theWire);
            
            if (i2c_dev == nullptr) {
                return false;
            }
            
            // Attempt to initialize the I2C device
            if (!i2c_dev->begin()) {
                delete i2c_dev;
                i2c_dev = nullptr;
                return false;
            }
            
            // Verify device presence by reading a known register
            uint8_t testBuffer;
            if (!i2c_dev->read(&testBuffer, 1)) {
                delete i2c_dev;
                i2c_dev = nullptr;
                return false;
            }
            
            initialized = true;
            return true;
            
        } catch (...) {
            // Handle any allocation failures
            if (i2c_dev != nullptr) {
                delete i2c_dev;
                i2c_dev = nullptr;
            }
            return false;
        }
    }
    
    // Get attachment status with proper error handling
    bool getAttached(bool* status = nullptr) {
        // Validate initialization state
        if (!initialized || i2c_dev == nullptr) {
            if (status != nullptr) {
                *status = false;
            }
            return false;  // Return false to indicate error
        }
        
        try {
            // Create register object with validation
            Adafruit_I2CRegister* attachRegister = 
                new Adafruit_I2CRegister(i2c_dev, HUSB238_PD_STATUS0, 1, MSBFIRST);
            
            if (attachRegister == nullptr) {
                if (status != nullptr) {
                    *status = false;
                }
                return false;
            }
            
            // Create bit object for the 7th bit
            Adafruit_I2CRegisterBit* attachBit = 
                new Adafruit_I2CRegisterBit(attachRegister, 7);
            
            if (attachBit == nullptr) {
                delete attachRegister;
                if (status != nullptr) {
                    *status = false;
                }
                return false;
            }
            
            // Read the bit with error checking
            bool attachStatus = false;
            bool readSuccess = false;
            
            // Attempt to read with timeout/retry logic
            const int MAX_RETRIES = 3;
            for (int retry = 0; retry < MAX_RETRIES; retry++) {
                try {
                    attachStatus = attachBit->read();
                    readSuccess = true;
                    break;
                } catch (...) {
                    // Continue to next retry
                    delay(10);  // Small delay between retries
                }
            }
            
            // Clean up allocated objects
            delete attachBit;
            delete attachRegister;
            
            if (readSuccess) {
                if (status != nullptr) {
                    *status = attachStatus;
                }
                return true;  // Success
            } else {
                if (status != nullptr) {
                    *status = false;
                }
                return false;  // Read failed
            }
            
        } catch (...) {
            // Handle any exceptions during operation
            if (status != nullptr) {
                *status = false;
            }
            return false;
        }
    }
    
    // Check if device is initialized
    bool isInitialized() const {
        return initialized && (i2c_dev != nullptr);
    }
    
    // Reset the device connection
    void reset() {
        if (i2c_dev != nullptr) {
            delete i2c_dev;
            i2c_dev = nullptr;
        }
        initialized = false;
    }
};