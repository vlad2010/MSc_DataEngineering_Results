static void write(uint16_t address, uint8_t value) {
    *reinterpret_cast<volatile uint8_t*>(address) = value;
}

static uint8_t read(uint16_t address) {
    return *reinterpret_cast<volatile uint8_t*>(address);
}