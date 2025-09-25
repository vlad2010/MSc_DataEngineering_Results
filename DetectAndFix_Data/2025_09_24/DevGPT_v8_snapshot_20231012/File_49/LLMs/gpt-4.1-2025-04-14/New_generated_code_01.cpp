#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <string>
#include <regex>

enum class LogLevel {
    ERROR = 0,
    WARNING,
    INFO,
    DEBUG
};

class Logger {
public:
    Logger(const std::string& filename = "") : logLevel(LogLevel::DEBUG) {
        if (!filename.empty()) {
            // Validate filename: only allow alphanumeric, underscore, dash, and .log extension
            std::regex valid_filename("^[\\w\\-]+\\.log$");
            if (!std::regex_match(filename, valid_filename)) {
                std::cerr << "Invalid log file name: " << filename << std::endl;
                return;
            }
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
    void log(const T& message, const char* file, int line, const char* function, LogLevel level) {
        if (level <= logLevel) {
            std::lock_guard<std::mutex> lock(logMutex);
            std::ostream& output = logFile.is_open() ? logFile : std::cout;
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);

            output << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
                   // Optionally comment out the next line to avoid exposing internal details
                   << "[" << file << ":" << line << "] "
                   << "[" << function << "] "
                   << logLevelToString(level) << ": " << message << std::endl;
        }
    }

private:
    std::ofstream logFile;
    LogLevel logLevel;
    std::mutex logMutex;

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