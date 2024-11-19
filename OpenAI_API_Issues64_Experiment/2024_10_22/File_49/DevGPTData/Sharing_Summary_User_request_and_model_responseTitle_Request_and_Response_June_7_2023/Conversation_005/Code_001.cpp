#include <iostream>
#include <fstream>
#include <ctime>

enum class LogLevel {
    NONE,
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

class Logger {
public:
    Logger(const std::string& filename, LogLevel level) : logFile(filename, std::ios::app), logLevel(level) {
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
    void log(const T& message, const char* file, int line, LogLevel level) {
        if (logFile.is_open() && level <= logLevel) {
            std::time_t currentTime = std::time(nullptr);
            logFile << "[" << std::asctime(std::localtime(&currentTime)) << "] "
                    << "[" << file << ":" << line << "] " << logLevelToString(level) << ": " << message << std::endl;
        }
    }

private:
    std::ofstream logFile;
    LogLevel logLevel;

    std::string logLevelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::ERROR:
                return "ERROR";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::DEBUG:
                return "DEBUG";
            default:
                return "";
        }
    }
};
