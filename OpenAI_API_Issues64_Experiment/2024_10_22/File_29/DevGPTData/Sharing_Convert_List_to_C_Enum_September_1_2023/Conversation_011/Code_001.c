typedef enum _husb_currents {
    CURRENT_0_5_A = 0b0000,  ///< 0.5A
    CURRENT_0_7_A = 0b0001,  ///< 0.7A
    CURRENT_1_0_A = 0b0010,  ///< 1.0A
    CURRENT_1_25_A = 0b0011, ///< 1.25A
    CURRENT_1_5_A = 0b0100,  ///< 1.5A
    CURRENT_1_75_A = 0b0101, ///< 1.75A
    CURRENT_2_0_A = 0b0110,  ///< 2.0A
    CURRENT_2_25_A = 0b0111, ///< 2.25A
    CURRENT_2_50_A = 0b1000, ///< 2.50A
    CURRENT_2_75_A = 0b1001, ///< 2.75A
    CURRENT_3_0_A = 0b1010,  ///< 3.0A
    CURRENT_3_25_A = 0b1011, ///< 3.25A
    CURRENT_3_5_A = 0b1100,  ///< 3.5A
    CURRENT_4_0_A = 0b1101,  ///< 4.0A
    CURRENT_4_5_A = 0b1110,  ///< 4.5A
    CURRENT_5_0_A = 0b1111   ///< 5.0A
} HUSB238_CurrentSetting;
