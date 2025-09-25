#include <pcre.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <memory>
#include <limits>

bool matchRegex(pcre* re, const char* subject) {
    // CWE-476 Fix: Validate input parameters
    if (re == nullptr) {
        std::cerr << "Error: Invalid regex object (null)" << std::endl;
        return false;
    }
    
    if (subject == nullptr) {
        std::cerr << "Error: Invalid subject string (null)" << std::endl;
        return false;
    }
    
    // CWE-20 Fix: Validate subject length and prevent integer overflow
    size_t subjectLength = strlen(subject);
    if (subjectLength > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "Error: Subject string too long" << std::endl;
        return false;
    }
    
    // CWE-120 Fix: Query the regex for capture count and allocate appropriately
    int captureCount = 0;
    int infoResult = pcre_fullinfo(re, nullptr, PCRE_INFO_CAPTURECOUNT, &captureCount);
    if (infoResult != 0) {
        std::cerr << "Error: Failed to get regex information" << std::endl;
        return false;
    }
    
    // Allocate ovector dynamically based on actual needs
    // PCRE requires (captureCount + 1) * 3 integers
    int ovectorSize = (captureCount + 1) * 3;
    
    // Protect against excessive memory allocation
    const int MAX_CAPTURE_GROUPS = 100;
    if (captureCount > MAX_CAPTURE_GROUPS) {
        std::cerr << "Error: Too many capture groups in regex" << std::endl;
        return false;
    }
    
    // Use vector for automatic memory management
    std::vector<int> ovector(ovectorSize);
    
    // Perform the regex match with validated inputs
    int rc = pcre_exec(
        re, 
        nullptr, 
        subject, 
        static_cast<int>(subjectLength), 
        0, 
        0, 
        ovector.data(), 
        ovectorSize
    );
    
    // CWE-209 Fix: Sanitize error messages
    if (rc >= 0) {
        std::cout << "Match found!" << std::endl;
        return true;
    } else if (rc == PCRE_ERROR_NOMATCH) {
        std::cout << "No match found." << std::endl;
    } else {
        // Don't expose specific error codes to prevent information leakage
        switch(rc) {
            case PCRE_ERROR_NULL:
                std::cerr << "Error: Invalid regex parameters" << std::endl;
                break;
            case PCRE_ERROR_BADOPTION:
                std::cerr << "Error: Invalid regex options" << std::endl;
                break;
            case PCRE_ERROR_BADMAGIC:
            case PCRE_ERROR_UNKNOWN_OPCODE:
            case PCRE_ERROR_INTERNAL:
                std::cerr << "Error: Corrupted regex object" << std::endl;
                break;
            case PCRE_ERROR_NOMEMORY:
                std::cerr << "Error: Insufficient memory for regex operation" << std::endl;
                break;
            case PCRE_ERROR_MATCHLIMIT:
            case PCRE_ERROR_RECURSIONLIMIT:
                std::cerr << "Error: Regex complexity limit exceeded" << std::endl;
                break;
            default:
                // Generic error message for unknown errors
                std::cerr << "Error: Regex matching failed" << std::endl;
                break;
        }
    }
    
    return false;
}