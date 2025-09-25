#include <SoftI2C.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/*
Optional locking callbacks to make the function thread/ISR-safe.
If your environment is single-threaded, leave them as NULL.
In an RTOS, set them to take/release a mutex or disable/enable interrupts.
*/
typedef void (*i2c_lock_fn_t)(void);
static i2c_lock_fn_t s_i2c_lock   = NULL;
static i2c_lock_fn_t s_i2c_unlock = NULL;

void Wire_setI2CLockCallbacks(i2c_lock_fn_t lock_fn, i2c_lock_fn_t unlock_fn) {
    s_i2c_lock = lock_fn;
    s_i2c_unlock = unlock_fn;
}

static inline bool is_valid_7bit_unreserved_addr(uint8_t addr7)
{
    // Valid, non-reserved 7-bit I2C addresses: 0x08..0x77
    // 0x00..0x07 and 0x78..0x7F are reserved per I2C spec
    return (addr7 >= 0x08) && (addr7 <= 0x77);
}

/*
Fixed version:
- Validates address against reserved ranges (CWE-20).
- Masks to 7 bits before shifting to avoid malformed address byte (CWE-190).
- Validates data pointer when bytes > 0 (CWE-476).
- Adds optional locking hooks to mitigate race conditions (CWE-362).
- Bounds the number of retries on the address phase to avoid indefinite blocking if the device NACKs (helps mitigate CWE-400 in some cases).
Note: If SoftI2C provides timeout-aware APIs, prefer using those and check their return values.
*/
bool Wire_writeBytes(uint8_t i2caddr, const uint8_t *data, uint8_t bytes)
{
    // Input validation
    uint8_t addr7 = (uint8_t)(i2caddr & 0x7F);
    if (!is_valid_7bit_unreserved_addr(addr7)) {
        // Reject reserved/out-of-range addresses by default for safety.
        return false;
    }
    if (bytes > 0 && data == NULL) {
        // Avoid null dereference
        return false;
    }

    if (s_i2c_lock) s_i2c_lock();

    bool ok = false; // overall result
    uint8_t addrByte = (uint8_t)((addr7 << 1) | 0x00); // write bit = 0
    uint8_t ack_bit;

    // Optional bounded retry on address phase to handle transient NACKs
    const uint8_t max_addr_retries = 3;
    uint8_t attempt = 0;

    // Start + address phase with retries
    for (attempt = 0; attempt < max_addr_retries; ++attempt) {
        I2CStart();

        ack_bit = I2CSend(addrByte);
        if (ack_bit == 0) {
            // Address ACKed, proceed to data phase
            break;
        }

        // NACK on address: stop, brief backoff (if platform has delay), retry
        I2CStop();
        // If you have a platform delay, add it here (e.g., delayMicroseconds(50))
    }

    if (attempt == max_addr_retries) {
        // Could not address the device
        if (s_i2c_unlock) s_i2c_unlock();
        return false;
    }

    // Data phase
    for (uint8_t i = 0; i < bytes; i++) {
        ack_bit = I2CSend(data[i]);
        if (ack_bit != 0) {
            // NACK on data byte
            I2CStop();
            if (s_i2c_unlock) s_i2c_unlock();
            return false;
        }
    }

    I2CStop();
    ok = true;

    if (s_i2c_unlock) s_i2c_unlock();
    return ok;
}