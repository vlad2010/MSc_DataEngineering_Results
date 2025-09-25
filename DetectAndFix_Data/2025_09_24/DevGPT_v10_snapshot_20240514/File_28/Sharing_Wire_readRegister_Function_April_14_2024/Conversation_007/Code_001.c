bool Wire_scan(uint8_t i2caddr) {
    // No actual data bytes to send, just the address
    return Wire_writeBytes(i2caddr, NULL, 0);
}