#include <SoftI2C.h>

bool Wire_writeBytes(uint8_t i2caddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;

    I2CStart();
    ack_bit = I2CSend(i2caddr << 1 | 0); // Shift address and append write bit
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }

    for (uint8_t i = 0; i < bytes; i++) {
        if (I2CSend(data[i]) != 0) {
            I2CStop();
            return false;
        }
    }

    I2CStop();
    return true;
}