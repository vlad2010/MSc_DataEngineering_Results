typedef enum {
  NORMAL_MODE = 0x00,             // PD1 = 0, PD0 = 0
  K1_OUTPUT_IMPEDANCE = 0x01,     // PD1 = 0, PD0 = 1
  K100_OUTPUT_IMPEDANCE = 0x02,   // PD1 = 1, PD0 = 0
  THREE_STATE_OUTPUT_IMPEDANCE = 0x03  // PD1 = 1, PD0 = 1
} ad5693_operating_modes;
