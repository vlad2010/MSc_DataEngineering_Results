#include <SoftI2C.h>

extern uint8_t scl_pin;
extern uint8_t sda_pin;

bool Wire_scan(uint8_t i2caddr) {
    uint8_t ack_bit;

    I2CStart();
    ack_bit = I2CSend((i2caddr << 1) | 0); // Trying to write to the address
    I2CStop();

    return (ack_bit == 0); // Return true if device acknowledged the address
}