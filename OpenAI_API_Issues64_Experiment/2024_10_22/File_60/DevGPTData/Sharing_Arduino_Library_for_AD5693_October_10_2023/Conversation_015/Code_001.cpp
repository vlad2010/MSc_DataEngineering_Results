typedef enum {
  NORMAL_MODE = 0x00,             // PD1 = 0, PD0 = 0
  OUTPUT_1K_IMPEDANCE = 0x01,     // PD1 = 0, PD0 = 1
  OUTPUT_100K_IMPEDANCE = 0x02,   // PD1 = 1, PD0 = 0
  OUTPUT_TRISTATE = 0x03          // PD1 = 1, PD0 = 1
} ad5693_operating_modes;
