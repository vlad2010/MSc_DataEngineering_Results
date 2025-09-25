    class UltrasonicSensor {
    private:
        // HC-04 variables
        uint8_t _trigPin;
        uint8_t _echoPin;
        long _duration;
        char _id[3]; // Assuming IDs are two characters plus null terminator

    public:
        // Constructor
        UltrasonicSensor(const char* id) {
            strncpy(_id, id, sizeof(_id) - 1);
            _id[sizeof(_id) - 1] = '\0';
        }
        
        // ... rest of the class
    };