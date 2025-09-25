#include <iomanip>
#include <sstream>
#include <ctime>
#include <mutex>

void format_timestamp(std::ostream& os, uint64_t timestamp) {
    std::time_t time_t = timestamp / 1000000;
    
    // Use thread-safe gmtime_r on POSIX or gmtime_s on Windows
    struct tm tm_result;
    struct tm* gmtime_ptr = nullptr;
    
#ifdef _WIN32
    // Windows version
    errno_t err = gmtime_s(&tm_result, &time_t);
    if (err == 0) {
        gmtime_ptr = &tm_result;
    }
#else
    // POSIX version
    gmtime_ptr = gmtime_r(&time_t, &tm_result);
#endif
    
    if (gmtime_ptr == nullptr) {
        // Handle error case - invalid timestamp
        os << "[Invalid timestamp: " << timestamp << "]";
        return;
    }
    
    // Use larger buffer to be safe and fix format string typo
    char buffer[64];
    size_t result = strftime(buffer, sizeof(buffer), "%Y-%m-%d %T.", gmtime_ptr);
    
    if (result == 0) {
        // strftime failed
        os << "[Format error: " << timestamp << "]";
        return;
    }
    
    // Calculate microseconds safely
    uint32_t microseconds = static_cast<uint32_t>(timestamp % 1000000);
    
    // Use stream manipulators for safe formatting
    os << '[' << buffer 
       << std::setfill('0') << std::setw(6) << microseconds 
       << ']';
}