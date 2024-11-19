#include <iostream>
#include <fstream>
#include <ctime>

class Logger {
public:
    Logger(const std::string& filename) : logFile(filename, std::ios::app) {
        if (!logFile.is_open()) {
            std::cerr << "Error opening log file: " << filename << std::endl;
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    template<typename T>
    void log(const T& message, const char* file, int line) {
        if (logFile.is_open()) {
            std::time_t currentTime = std::time(nullptr);
            logFile << "[" << std::asctime(std::localtime(&currentTime)) << "] "
                    << "[" << file << ":" << line << "] " << message << std::endl;
        }
    }

private:
    std::ofstream logFile;
};
