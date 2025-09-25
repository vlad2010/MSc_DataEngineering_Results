#include "Adafruit_HUSB238.h"
#include <memory> // For std::unique_ptr

class Adafruit_HUSB238 {
public:
    Adafruit_HUSB238();
    ~Adafruit_HUSB238();
    bool begin(uint8_t addr, TwoWire *theWire = &Wire);
    bool getCCStatus();
    bool getAttached();

private:
    std::unique_ptr<Adafruit_I2CDevice> i2c_dev;
};

Adafruit_HUSB238::Adafruit_HUSB238() : i2c_dev(nullptr) {}

Adafruit_HUSB238::~Adafruit_HUSB238() {
    // unique_ptr will automatically delete i2c_dev
}

bool Adafruit_HUSB238::begin(uint8_t addr, TwoWire *theWire) {
    // Use unique_ptr for automatic memory management
    i2c_dev = std::make_unique<Adafruit_I2CDevice>(addr, theWire);

    /* Try to instantiate the I2C device. */
    if (!i2c_dev->begin()) {
        i2c_dev.reset(); // Ensure pointer is null if begin fails
        return false;
    }

    return true;
}

bool Adafruit_HUSB238::getCCStatus() {
    if (!i2c_dev) {
        // Log error or handle as needed
        return false;
    }
    Adafruit_I2CRegister ccRegister(i2c_dev.get(), HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit ccBit(&ccRegister, 7);  // 7th bit
    return ccBit.read();
}

bool Adafruit_HUSB238::getAttached() {
    if (!i2c_dev) {
        // Log error or handle as needed
        return false;
    }
    Adafruit_I2CRegister attachRegister(i2c_dev.get(), HUSB238_PD_STATUS1);
    Adafruit_I2CRegisterBit attachBit(&attachRegister, 6);  // 6th bit
    return attachBit.read();
}

// ... (continue with the rest of the functions in a similar manner)