#include <SoftI2C.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Define maximum reasonable bytes for I2C transaction
#define MAX_I2C_TRANSACTION_SIZE 255
// Valid I2C address range (7-bit addressing)
#define MAX_I2C_ADDRESS 0x7F
#define MIN_I2C_ADDRESS 0x08  // Addresses 0x00-0x07 are reserved

bool Wire_writeBytes(uint8_t i2caddr, uint8_t *data, uint8_t bytes) {
    uint8_t ack_bit;
    
    // Input validation - CWE-20 fix
    // Validate I2C address range (7-bit addressing)
    if (i2caddr > MAX_I2C_ADDRESS || i2caddr < MIN_I2C_ADDRESS) {
        // Invalid I2C address
        return false;
    }
    
    // Null pointer check - CWE-476 fix
    if (data == NULL) {
        return false;
    }
    
    // Validate bytes parameter - CWE-20 fix
    if (bytes == 0 || bytes > MAX_I2C_TRANSACTION_SIZE) {
        return false;
    }
    
    // Start I2C transaction
    I2CStart();
    
    // Send address with write bit (0)
    // Safe because we validated i2caddr <= 0x7F, so shift won't overflow
    ack_bit = I2CSend((uint8_t)((i2caddr << 1) | 0));
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }
    
    // Send data bytes
    for (uint8_t i = 0; i < bytes; i++) {
        // At this point, data is guaranteed non-NULL and 
        // caller is responsible for ensuring buffer size
        if (I2CSend(data[i]) != 0) {
            I2CStop();
            return false;
        }
    }
    
    I2CStop();
    return true;
}

// Optional: Safer version with explicit buffer size parameter
bool Wire_writeBytesSecure(uint8_t i2caddr, uint8_t *data, size_t buffer_size, uint8_t bytes_to_write) {
    uint8_t ack_bit;
    
    // Input validation
    if (i2caddr > MAX_I2C_ADDRESS || i2caddr < MIN_I2C_ADDRESS) {
        return false;
    }
    
    if (data == NULL) {
        return false;
    }
    
    if (bytes_to_write == 0 || bytes_to_write > MAX_I2C_TRANSACTION_SIZE) {
        return false;
    }
    
    // CWE-823 fix: Ensure we don't read beyond buffer bounds
    if (bytes_to_write > buffer_size) {
        return false;
    }
    
    I2CStart();
    
    ack_bit = I2CSend((uint8_t)((i2caddr << 1) | 0));
    if (ack_bit != 0) {
        I2CStop();
        return false;
    }
    
    for (uint8_t i = 0; i < bytes_to_write; i++) {
        if (I2CSend(data[i]) != 0) {
            I2CStop();
            return false;
        }
    }
    
    I2CStop();
    return true;
}