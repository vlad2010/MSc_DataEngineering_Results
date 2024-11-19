typedef enum _husb_pdo_selection {
    PDO_NOT_SELECTED = 0b0000, ///< Not selected
    PDO_SRC_5V = 0b0001,       ///< SRC_PDO_5V
    PDO_SRC_9V = 0b0010,       ///< SRC_PDO_9V
    PDO_SRC_12V = 0b0011,      ///< SRC_PDO_12V
    PDO_SRC_15V = 0b1000,      ///< SRC_PDO_15V
    PDO_SRC_18V = 0b1001,      ///< SRC_PDO_18V
    PDO_SRC_20V = 0b1010       ///< SRC_PDO_20V
} HUSB238_PDOSelection;
