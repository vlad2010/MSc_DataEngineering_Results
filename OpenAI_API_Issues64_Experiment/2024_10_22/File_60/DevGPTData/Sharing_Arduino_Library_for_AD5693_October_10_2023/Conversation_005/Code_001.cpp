enum AD5693_Commands {
  NOP = 0x00,            // NOP: do nothing
  WRITE_INPUT = 0x10,    // Write input register
  UPDATE_DAC = 0x20,     // Update DAC register (LDAC software)
  WRITE_DAC_AND_INPUT = 0x30, // Write DAC and input registers
  WRITE_CONTROL = 0x40   // Write control register
};
