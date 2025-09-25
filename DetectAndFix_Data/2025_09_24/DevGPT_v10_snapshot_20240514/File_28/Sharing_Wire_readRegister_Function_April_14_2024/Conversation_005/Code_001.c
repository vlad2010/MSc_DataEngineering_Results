#include <SoftI2C.h>
#include <stdbool.h>
#include <stdint.h>

// Constants for the AHT20
#define AHTX0_I2CADDR_DEFAULT 0x38
#define AHTX0_CMD_TRIGGER 0xAC
#define AHTX0_STATUS_BUSY 0x80

bool AHT20_getStatus(uint8_t *status) {
    I2CStart();
    if (I2CSend(AHTX0_I2CADDR_DEFAULT << 1 | 1) == 0) { // Read operation
        *status = I2CRead();
        I2CNak();
        I2CStop();
        return true;
    }
    I2CStop();
    return false;
}

bool readAHT20(float *temperature, float *humidity) {
    uint8_t cmd[3] = {AHTX0_CMD_TRIGGER, 0x33, 0x00};
    uint8_t data[6], status;
    uint32_t rawHumidity, rawTemperature;

    I2CStart();
    if (I2CSend(AHTX0_I2CADDR_DEFAULT << 1 | 0) != 0 || // Write address
        I2CSend(cmd[0]) != 0 ||
        I2CSend(cmd[1]) != 0 ||
        I2CSend(cmd[2]) != 0) {
        I2CStop();
        return false;
    }
    I2CStop();

    do {
        if (!AHT20_getStatus(&status)) {
            return false;
        }
        delay(10); // Delay 10ms to wait for measurement
    } while (status & AHTX0_STATUS_BUSY);

    I2CStart();
    if (I2CSend(AHTX0_I2CADDR_DEFAULT << 1 | 1) != 0) { // Read operation
        I2CStop();
        return false;
    }

    for (int i = 0; i < 6; i++) {
        data[i] = I2CRead();
        if (i < 5) I2CAck();
    }
    I2CNak();
    I2CStop();

    // Parse humidity data
    rawHumidity = data[1];
    rawHumidity = (rawHumidity << 8) | data[2];
    rawHumidity = (rawHumidity << 4) | (data[3] >> 4);
    *humidity = ((float)rawHumidity * 100.0) / 0x100000;

    // Parse temperature data
    rawTemperature = (data[3] & 0x0F);
    rawTemperature = (rawTemperature << 8) | data[4];
    rawTemperature = (rawTemperature << 8) | data[5];
    *temperature = ((float)rawTemperature * 200.0 / 0x100000) - 50.0;

    return true;
}