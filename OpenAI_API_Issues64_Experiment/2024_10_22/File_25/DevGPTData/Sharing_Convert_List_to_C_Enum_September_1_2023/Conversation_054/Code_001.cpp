class Adafruit_HUSB238 {
  // ... (other members and constructor)
  
  public:
    // ... (other functions)
    
    // Getter functions
    bool isVoltageDetected(HUSB238_PDOSelection pd);
    HUSB238_CurrentSetting currentDetected(HUSB238_PDOSelection pd);
    
    // Setter functions
    void selectPDO(HUSB238_PDOSelection pd);
    void reset();
    void requestPDO();
    void getSourceCapabilities();
    
    // ... (other functions)
};
