class UltrasonicSensor {
private:
    // HC-04 variables
    uint8_t _trigPin;
    uint8_t _echoPin;
    long _duration;
    char _id[3]; // Assuming IDs are two characters plus null terminator

public:
    // Constructor
    UltrasonicSensor(const char id[3]) {
        // Copy characters from the provided id to the _id array
        strncpy(_id, id, sizeof(_id) - 1);
        // Ensure null termination
        _id[sizeof(_id) - 1] = '\0';
    }

    // ... rest of the class
};