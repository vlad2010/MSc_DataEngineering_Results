#include <algorithm>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

// Configuration flags (should be set from environment or config file)
#ifndef NDEBUG
    constexpr bool DEBUG_MODE = true;
    constexpr bool LOG_SENSITIVE_DATA = false; // Even in debug, don't log sensitive data by default
#else
    constexpr bool DEBUG_MODE = false;
    constexpr bool LOG_SENSITIVE_DATA = false;
#endif

constexpr size_t MAX_LOG_LENGTH = 100; // Maximum characters to log

// Sanitize string for safe logging (remove control characters)
std::string sanitize_for_logging(const std::string& input, size_t max_length = MAX_LOG_LENGTH) {
    std::stringstream sanitized;
    size_t count = 0;
    
    for (char c : input) {
        if (count >= max_length) {
            sanitized << "...[truncated]";
            break;
        }
        
        // Only allow printable ASCII characters (0x20 to 0x7E)
        if (c >= 0x20 && c <= 0x7E) {
            sanitized << c;
        } else {
            // Replace non-printable characters with their hex representation
            sanitized << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<unsigned int>(static_cast<unsigned char>(c));
        }
        count++;
    }
    
    return sanitized.str();
}

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
    // Input validation
    if (ptr == nullptr || data == nullptr) {
        return 0;
    }
    
    if (data->empty()) {
        return 0;
    }

    // Prevent integer overflow in size calculation
    size_t total_size = size * nmemb;
    if (size != 0 && total_size / size != nmemb) {
        // Integer overflow detected
        return 0;
    }

    size_t len = std::min(total_size, data->size());
    
    // Safe memory copy
    std::memcpy(ptr, data->data(), len);
    data->erase(0, len);

    // Conditional debug logging with security considerations
    if (DEBUG_MODE) {
        // Log metadata instead of actual content
        std::cerr << "[DEBUG] Read callback: transferred " << len << " bytes" << std::endl;
        
        // Only log actual data if explicitly enabled and in debug mode
        // This should NEVER be enabled in production
        if (LOG_SENSITIVE_DATA) {
            // Create a safe copy for logging
            std::string log_data(static_cast<char*>(ptr), std::min(len, MAX_LOG_LENGTH));
            std::string sanitized = sanitize_for_logging(log_data);
            
            std::cerr << "[DEBUG] Body preview (sanitized): " << sanitized << std::endl;
            
            // Add warning about sensitive data logging
            static bool warning_shown = false;
            if (!warning_shown) {
                std::cerr << "[WARNING] Sensitive data logging is enabled. "
                         << "This should be disabled in production!" << std::endl;
                warning_shown = true;
            }
        }
    }

    return len;
}