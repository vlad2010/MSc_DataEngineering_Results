#include <iostream>
#include <fstream>
#include <ctime>
#include <mutex>
#include <iomanip>
#include <sstream>

enum class LogLevel {
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

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
        std::lock_guard<std::mutex> lock(logMutex);
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    template<typename T>
    void log(const T& message, const char* file, int line, const char* function, LogLevel level) {
        if (level <= logLevel) {
            std::lock_guard<std::mutex> lock(logMutex);  // Ensure thread safety
            std::ostream& output = logFile.is_open() ? logFile : std::cout;

            auto currentTime = std::time(nullptr);
            std::tm tm = *std::localtime(&currentTime);
            std::ostringstream timeStream;
            timeStream << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

            output << "[" << timeStream.str() << "] "
                   << "[" << file << ":" << line << "] "
                   << "[" << function << "] "
                   << logLevelToString(level) << ": " << message << std::endl;
        }
    }

private:
    std::ofstream logFile;
    LogLevel logLevel;
    std::mutex logMutex;  // Add a mutex to protect shared resources

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