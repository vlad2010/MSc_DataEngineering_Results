#include <cstdint>
#include <exception>

// Strongly typed enum for PDO selection
enum class HUSB238_PDOSelection : uint8_t {
    PDO_5V,
    PDO_9V,
    PDO_12V,
    PDO_15V,
    PDO_18V,
    PDO_20V,
    PDO_INVALID
};

constexpr uint8_t HUSB238_SRC_PDO_5V  = 0x01;
constexpr uint8_t HUSB238_SRC_PDO_9V  = 0x02;
constexpr uint8_t HUSB238_SRC_PDO_12V = 0x03;
constexpr uint8_t HUSB238_SRC_PDO_15V = 0x04;
constexpr uint8_t HUSB238_SRC_PDO_18V = 0x05;
constexpr uint8_t HUSB238_SRC_PDO_20V = 0x06;
constexpr uint8_t PDO_STATUS_BIT = 7; // 7th bit

bool Adafruit_HUSB238::isVoltageDetected(HUSB238_PDOSelection pd) {
    uint8_t registerAddress = 0;

    // Validate input and determine register address
    switch(pd) {
        case HUSB238_PDOSelection::PDO_5V:
            registerAddress = HUSB238_SRC_PDO_5V;
            break;
        case HUSB238_PDOSelection::PDO_9V:
            registerAddress = HUSB238_SRC_PDO_9V;
            break;
        case HUSB238_PDOSelection::PDO_12V:
            registerAddress = HUSB238_SRC_PDO_12V;
            break;
        case HUSB238_PDOSelection::PDO_15V:
            registerAddress = HUSB238_SRC_PDO_15V;
            break;
        case HUSB238_PDOSelection::PDO_18V:
            registerAddress = HUSB238_SRC_PDO_18V;
            break;
        case HUSB238_PDOSelection::PDO_20V:
            registerAddress = HUSB238_SRC_PDO_20V;
            break;
        default:
            // Invalid PDO selection
            return false;
    }

    try {
        // Create an Adafruit_I2CRegister object for the selected register
        Adafruit_I2CRegister pdoRegister(i2c_dev, registerAddress);

        // Create an Adafruit_I2CRegisterBit object for the status bit
        Adafruit_I2CRegisterBit pdoBit(&pdoRegister, PDO_STATUS_BIT);

        // Read and return the bit, handle read errors
        int bitValue = pdoBit.read();
        if (bitValue < 0) {
            // Read failed, do not leak details
            return false;
        }
        return bitValue != 0;
    } catch (const std::exception&) {
        // Catch any exceptions, do not leak details
        return false;
    }
}