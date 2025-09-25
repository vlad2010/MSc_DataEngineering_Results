#include <SoftI2C.h>

// Assume scl_pin and sda_pin are declared as extern in SoftI2C.h and initialized elsewhere
extern uint8_t scl_pin;
extern uint8_t sda_pin;

void Wire_readRegister(uint8_t i2caddr, uint8_t regaddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;

    I2CStart();
    ack_bit = I2CSend((i2caddr << 1) | 0); // Write operation
    if (ack_bit != 0) {
        I2CStop();
        return;
    }

    // Send the register address from where data will be read
    ack_bit = I2CSend(regaddr);
    if (ack_bit != 0) {
        I2CStop();
        return;
    }

    I2CRestart();
    ack_bit = I2CSend((i2caddr << 1) | 1); // Read operation
    if (ack_bit != 0) {
        I2CStop();
        return;
    }

    // Read the specified number of bytes from the EEPROM
    for (uint8_t i = 0; i < bytes; i++) {
        data[i] = I2CRead();
        if (i == (bytes - 1)) {
            I2CNak();  // Send NAK after the last byte
        } else {
            I2CAck();  // Send ACK after reading a byte
        }
    }

    I2CStop();
}