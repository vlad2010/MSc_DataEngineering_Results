#include <cstring>
#include <stdexcept>
#include <cstdint>

class UltrasonicSensor {
private:
    // HC-04 variables
    uint8_t _trigPin;
    uint8_t _echoPin;
    long _duration;
    char _id[3]; // Assuming IDs are two characters plus null terminator

public:
    // Constructor with proper initialization and validation
    UltrasonicSensor(const char* id, uint8_t trigPin = 0, uint8_t echoPin = 0) 
        : _trigPin(trigPin), _echoPin(echoPin), _duration(0) {
        
        // Validate input pointer
        if (id == nullptr) {
            throw std::invalid_argument("ID cannot be null");
        }
        
        // Check the length of the input string
        size_t idLength = strnlen(id, 3);
        
        // Validate that ID is exactly 2 characters (plus null terminator)
        if (idLength > 2) {
            throw std::invalid_argument("ID must be at most 2 characters");
        }
        
        // Clear the buffer first (defense in depth)
        memset(_id, 0, sizeof(_id));
        
        // Safe copy with guaranteed null termination
        strncpy(_id, id, sizeof(_id) - 1);
        _id[sizeof(_id) - 1] = '\0'; // Ensure null termination
    }
    
    // Alternative constructor using std::string for better safety
    explicit UltrasonicSensor(const std::string& id, uint8_t trigPin = 0, uint8_t echoPin = 0)
        : _trigPin(trigPin), _echoPin(echoPin), _duration(0) {
        
        // Validate string length
        if (id.length() > 2) {
            throw std::invalid_argument("ID must be at most 2 characters");
        }
        
        // Clear the buffer
        memset(_id, 0, sizeof(_id));
        
        // Safe copy from std::string
        id.copy(_id, sizeof(_id) - 1);
        _id[sizeof(_id) - 1] = '\0'; // Ensure null termination
    }
    
    // Getter methods for safe access
    const char* getId() const { return _id; }
    uint8_t getTrigPin() const { return _trigPin; }
    uint8_t getEchoPin() const { return _echoPin; }
    
    // Setter methods with validation
    void setTrigPin(uint8_t pin) { _trigPin = pin; }
    void setEchoPin(uint8_t pin) { _echoPin = pin; }
    
    // ... rest of the class implementation
};