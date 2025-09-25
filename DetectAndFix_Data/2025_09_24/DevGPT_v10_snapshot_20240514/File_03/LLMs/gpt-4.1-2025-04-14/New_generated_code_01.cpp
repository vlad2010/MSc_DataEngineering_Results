#include <cstdint>
#include <cstring>
#include <stdexcept>

class UltrasonicSensor {
private:
    // HC-04 variables
    uint8_t _trigPin = 0;
    uint8_t _echoPin = 0;
    long _duration = 0;
    char _id[3]; // Two characters plus null terminator

public:
    // Constructor
    UltrasonicSensor(const char* id) {
        if (id == nullptr) {
            throw std::invalid_argument("id cannot be null");
        }
        // Ensure the input is at most 2 characters
        size_t len = std::strnlen(id, 3);
        if (len == 0) {
            throw std::invalid_argument("id cannot be empty");
        }
        if (len > 2) {
            throw std::invalid_argument("id must be at most 2 characters");
        }
        // Copy safely and null-terminate
        std::memset(_id, 0, sizeof(_id));
        std::memcpy(_id, id, len);
        _id[len] = '\0';
    }

    // ... rest of the class
};