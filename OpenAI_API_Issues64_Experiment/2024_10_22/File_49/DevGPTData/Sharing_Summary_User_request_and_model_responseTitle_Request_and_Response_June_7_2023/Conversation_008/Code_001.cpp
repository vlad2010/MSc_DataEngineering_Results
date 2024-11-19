#include <iostream>
#include <fstream>
#include <ctime>

class Logger {
public:
    Logger(const std::string& filename = "") : logLevel(LogLevel::DEBUG) {
        if (!filename.empty()) {
            logFile.open(filename, std::ios::app);
            if (!logFile.is_open()) {
                std::cerr << "Error opening log file: " << filename << std::endl;
            }
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    template<typename T>
    void log(const T& message, const char* file, int line, LogLevel level) {
        if (level <= logLevel) {
            std::ostream& output = logFile.is_open() ? logFile : std::cout;
            std::time_t currentTime = std::time(nullptr);
            output << "[" << std::asctime(std::localtime(&currentTime)) << "] "
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
