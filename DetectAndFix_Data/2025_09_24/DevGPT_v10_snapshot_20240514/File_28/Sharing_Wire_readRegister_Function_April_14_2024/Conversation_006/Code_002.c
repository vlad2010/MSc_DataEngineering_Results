bool Wire_readBytes(uint8_t i2caddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;

    I2CStart();
    ack_bit = I2CSend(i2caddr << 1 | 1); // Shift address and append read bit
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }

    for (uint8_t i = 0; i < bytes; i++) {
        data[i] = I2CRead();
        if (i == bytes - 1) {
            I2CNak();  // NAK on last byte
        } else {
            I2CAck();  // ACK on other bytes
        }
    }

    I2CStop();
    return true;
}