HUSB238_5VCurrentContract Adafruit_HUSB238::get5VContractA() {
  // Fail-closed default: choose the safest value (usually the lowest current contract).
  // If your enum defines an explicit "Unknown" or "Default/500mA" value, use that instead of 0.
  auto fail_closed = []() -> HUSB238_5VCurrentContract {
    return static_cast<HUSB238_5VCurrentContract>(0u);
  };

  // 1) Validate device pointer (CWE-476).
  if (i2c_dev == nullptr) {
    return fail_closed();
  }

  // 2) Read the register with explicit error checking (CWE-252).
  const uint8_t reg_addr = HUSB238_PD_STATUS0;
  uint8_t raw = 0;

  // Use a robust single transaction to avoid TOCTOU between write/read.
  if (!i2c_dev->write_then_read(&reg_addr, 1, &raw, 1)) {
    return fail_closed();
  }

  // Optional: read twice and compare to mitigate transient glitches.
  // uint8_t raw2 = 0;
  // if (!i2c_dev->write_then_read(&reg_addr, 1, &raw2, 1) || raw != raw2) {
  //   return fail_closed();
  // }

  // 3) Extract and validate the 2-bit field. Mask ensures 0..3.
  const uint8_t bits = static_cast<uint8_t>(raw & 0x03u);

  // 4) Convert with a safe cast (CWE-704).
  return static_cast<HUSB238_5VCurrentContract>(bits);
}