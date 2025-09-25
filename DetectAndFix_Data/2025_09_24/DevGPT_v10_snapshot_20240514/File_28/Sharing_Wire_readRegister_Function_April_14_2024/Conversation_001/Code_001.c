#include <stdint.h>
#include "CH552.h"
#include "I2C.h"

void Wire_readRegister(uint8_t i2caddr, uint8_t regaddr, uint8_t *data, uint8_t bytes) {
    uint8_t status;

    // Set start condition and send device address with write bit
    I2C_Start();
    status = I2C_SendByte((i2caddr << 1) | 0x00);  // Write operation
    if (status != I2C_ACK) {
        I2C_Stop();
        return;
    }

    // Send register address
    status = I2C_SendByte(regaddr);
    if (status != I2C_ACK) {
        I2C_Stop();
        return;
    }

    // Set start condition again (repeated start) and send device address with read bit
    I2C_Start();
    status = I2C_SendByte((i2caddr << 1) | 0x01);  // Read operation
    if (status != I2C_ACK) {
        I2C_Stop();
        return;
    }

    // Read bytes from I2C device
    while (bytes > 0) {
        if (bytes == 1) {
            *data = I2C_RecvByte(I2C_NACK);  // NACK after last byte
        } else {
            *data = I2C_RecvByte(I2C_ACK);
        }
        data++;
        bytes--;
    }

    // Set stop condition
    I2C_Stop();
}