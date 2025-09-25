#include <cstdint>
#include <cstring>

class UltrasonicSensor {
private:
    static constexpr std::size_t kIdLen = 2;               // two visible characters
    static constexpr char kDefaultId[kIdLen + 1] = "??";   // safe fallback

    uint8_t  _trigPin{0};
    uint8_t  _echoPin{0};
    uint32_t _duration{0};  // use fixed width; avoids surprises with long
    char     _id[kIdLen + 1]{};

    // Bounded length helper (avoids relying on non-portable strnlen)
    static std::size_t boundedLen(const char* s, std::size_t max) {
        if (!s) return 0;
        std::size_t i = 0;
        while (i < max && s[i] != '\0') { ++i; }
        return i;
    }

    void setIdUnchecked(const char* src) {
        // precondition: src != nullptr and length <= kIdLen
        std::size_t len = boundedLen(src, kIdLen);
        // copy only len bytes and always add terminator
        if (len > 0) {
            std::memcpy(_id, src, len);
        }
        _id[len] = '\0';
    }

public:
    // Default constructor sets safe defaults
    UltrasonicSensor() {
        std::memcpy(_id, kDefaultId, kIdLen + 1);
    }

    // Compile-time safe: only accepts exactly 2-character string literals (plus null)
    UltrasonicSensor(const char (&id)[kIdLen + 1]) {
        // id has exact size 3 (2 chars + '\0') enforced by the type system
        std::memcpy(_id, id, kIdLen + 1);
    }

    // Runtime-validated constructor: accepts pointer, validates length and null
    explicit UltrasonicSensor(const char* id) {
        if (!id) {
            // CWE-476 mitigation: handle null safely
            std::memcpy(_id, kDefaultId, kIdLen + 1);
            return;
        }
        // Enforce maximum length (no silent truncation)
        if (boundedLen(id, kIdLen + 1) > kIdLen) {
            // Too long: reject and use a safe default (policy choice)
            std::memcpy(_id, kDefaultId, kIdLen + 1);
        } else {
            setIdUnchecked(id);
        }
    }

    // Optional: runtime setter with validation
    bool setId(const char* id) {
        if (!id) return false;
        if (boundedLen(id, kIdLen + 1) > kIdLen) return false; // reject overly long IDs
        setIdUnchecked(id);
        return true;
    }

    // ... rest of the class
};