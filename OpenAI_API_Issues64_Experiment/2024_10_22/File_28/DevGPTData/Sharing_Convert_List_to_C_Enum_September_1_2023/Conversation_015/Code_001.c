typedef enum _husb_response_codes {
    NO_RESPONSE = 0b000,                  ///< No response
    SUCCESS = 0b001,                      ///< Success
    INVALID_CMD_OR_ARG = 0b011,           ///< Invalid command or argument
    CMD_NOT_SUPPORTED = 0b100,            ///< Command not supported
    TRANSACTION_FAIL_NO_GOOD_CRC = 0b101  ///< Transaction fail. No GoodCRC is received after sending
} HUSB238_ResponseCodes;
