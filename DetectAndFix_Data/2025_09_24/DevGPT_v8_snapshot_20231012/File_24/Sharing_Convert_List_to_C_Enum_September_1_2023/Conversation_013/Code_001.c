typedef enum _husb_voltages {
    UNATTACHED = 0b0000, ///< Unattached
    PD_5V = 0b0001,      ///< 5V
    PD_9V = 0b0010,      ///< 9V
    PD_12V = 0b0011,     ///< 12V
    PD_15V = 0b0100,     ///< 15V
    PD_18V = 0b0101,     ///< 18V
    PD_20V = 0b0110      ///< 20V
} HUSB238_VoltageSetting;
