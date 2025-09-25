bool Wire_readRegister(uint8_t i2caddr, uint8_t regaddr, uint8_t *data, uint8_t bytes) {
    // Write the register address first
    if (!Wire_writeBytes(i2caddr, &regaddr, 1)) {
        return false;
    }

    // Now read the bytes from that register
    return Wire_readBytes(i2caddr, data, bytes);
}