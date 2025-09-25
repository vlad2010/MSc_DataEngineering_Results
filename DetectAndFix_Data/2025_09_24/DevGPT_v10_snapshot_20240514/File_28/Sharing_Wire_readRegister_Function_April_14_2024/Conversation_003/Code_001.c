#include <SoftI2C.h>

extern uint8_t scl_pin;
extern uint8_t sda_pin;

bool Wire_readRegister(uint8_t i2caddr, uint8_t regaddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;

    I2CStart();
    ack_bit = I2CSend((i2caddr << 1) | 0);
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }

    ack_bit = I2CSend(regaddr);
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }

    I2CRestart();
    ack_bit = I2CSend((i2caddr << 1) | 1);
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }

    for (uint8_t i = 0; i < bytes; i++) {
        data[i] = I2CRead();
        if (i == (bytes - 1)) {
            I2CNak();
        } else {
            I2CAck();
        }
    }

    I2CStop();
    return true;
}