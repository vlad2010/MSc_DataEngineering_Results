#include <SoftI2C.h>
#include <stddef.h> // for NULL

#define MAX_I2C_WRITE_BYTES 128 // Set a reasonable upper limit

bool Wire_writeBytes(uint8_t i2caddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;

    // Security Fix 1: Check for NULL pointer
    if (data == NULL) {
        return false;
    }

    // Security Fix 2: Check for excessive write size
    if (bytes == 0 || bytes > MAX_I2C_WRITE_BYTES) {
        return false;
    }

    I2CStart();
    ack_bit = I2CSend((i2caddr << 1) | 0); // Shift address and append write bit
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