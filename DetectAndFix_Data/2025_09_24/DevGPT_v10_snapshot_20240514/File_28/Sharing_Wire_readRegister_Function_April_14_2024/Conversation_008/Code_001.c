#include <SoftI2C.h>

bool AHT20_getStatus(uint8_t *status) {
    return Wire_readBytes(AHTX0_I2CADDR_DEFAULT, status, 1);
}