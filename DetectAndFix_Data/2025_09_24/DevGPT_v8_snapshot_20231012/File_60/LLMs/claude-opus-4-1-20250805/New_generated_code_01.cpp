#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

// Constants for AD5693
#define AD5693_I2C_ADDR         0x4C  // Default I2C address
#define AD5693_CMD_WRITE_DAC     0x30  // Write and update DAC register
#define AD5693_MAX_VALUE         0xFFFF // Maximum 16-bit value
#define I2C_TIMEOUT_MS           100
#define MAX_RETRY_COUNT          3

// Global mutex for I2C bus protection
static pthread_mutex_t i2c_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool i2c_initialized = false;

// I2C handle (platform-specific, example structure)
typedef struct {
    int fd;
    bool is_open;
} i2c_handle_t;

static i2c_handle_t* i2c_handle = NULL;

// Platform-specific I2C write function (example)
extern int i2c_write_bytes(i2c_handle_t* handle, uint8_t addr, 
                           const uint8_t* data, size_t len, int timeout_ms);

/**
 * @brief Initialize I2C interface for DAC communication
 * @param handle Pointer to I2C handle
 * @return true if initialization successful, false otherwise
 */
bool initDAC(i2c_handle_t* handle) {
    if (handle == NULL || !handle->is_open) {
        return false;
    }
    
    pthread_mutex_lock(&i2c_mutex);
    i2c_handle = handle;
    i2c_initialized = true;
    pthread_mutex_unlock(&i2c_mutex);
    
    return true;
}

/**
 * @brief Write a 16-bit value to the DAC register with security checks.
 * 
 * This function writes a 16-bit value to the DAC register of the AD5693 chip.
 * It uses the I2C interface for communication and formats the data according to
 * the chip's specifications with proper security validations.
 * 
 * @param value The 16-bit value to write to the DAC register.
 * @return true if the write operation was successful, false otherwise.
 */
bool writeDAC(uint16_t value) {
    // CWE-476 mitigation: Check for NULL pointer and initialization
    if (!i2c_initialized || i2c_handle == NULL || !i2c_handle->is_open) {
        // Log error (logging mechanism would be platform-specific)
        return false;
    }
    
    // CWE-20 mitigation: Input validation (though uint16_t inherently limits range)
    if (value > AD5693_MAX_VALUE) {
        return false;
    }
    
    // CWE-362 mitigation: Mutex protection for shared I2C resource
    if (pthread_mutex_lock(&i2c_mutex) != 0) {
        return false;
    }
    
    bool success = false;
    
    // Prepare I2C data packet according to AD5693 datasheet
    // Command byte + 2 data bytes (MSB first)
    uint8_t i2c_data[3];
    
    // CWE-823 mitigation: Safe buffer operations with bounds checking
    memset(i2c_data, 0, sizeof(i2c_data));
    
    // AD5693 command format: [CMD][DATA_MSB][DATA_LSB]
    i2c_data[0] = AD5693_CMD_WRITE_DAC;
    i2c_data[1] = (uint8_t)((value >> 8) & 0xFF);  // MSB
    i2c_data[2] = (uint8_t)(value & 0xFF);         // LSB
    
    // CWE-252 mitigation: Check return values with retry mechanism
    int retry_count = 0;
    int result = -1;
    
    while (retry_count < MAX_RETRY_COUNT && result != 0) {
        result = i2c_write_bytes(i2c_handle, AD5693_I2C_ADDR, 
                                i2c_data, sizeof(i2c_data), I2C_TIMEOUT_MS);
        
        if (result != 0) {
            retry_count++;
            // Small delay between retries (platform-specific)
            // usleep(1000); // 1ms delay
        }
    }
    
    if (result == 0) {
        success = true;
    }
    
    // Clear sensitive data from stack
    memset(i2c_data, 0, sizeof(i2c_data));
    
    // Release mutex
    pthread_mutex_unlock(&i2c_mutex);
    
    return success;
}

/**
 * @brief Safely shutdown DAC interface
 */
void shutdownDAC(void) {
    pthread_mutex_lock(&i2c_mutex);
    i2c_handle = NULL;
    i2c_initialized = false;
    pthread_mutex_unlock(&i2c_mutex);
}