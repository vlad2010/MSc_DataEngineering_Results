    // Original
    UltrasonicSensor(String id) {
        _id = id;
    }

    // Modified
    UltrasonicSensor(const char* id) {
        strncpy(_id, id, sizeof(_id) - 1);
        _id[sizeof(_id) - 1] = '\0';
    }