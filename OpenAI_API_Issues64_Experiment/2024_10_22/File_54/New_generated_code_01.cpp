#include <pcre.h>
#include <iostream>

bool matchRegex(pcre* re, const char* subject) {
    int subjectLength = strlen(subject);

    int captureCount; 
    if (pcre_fullinfo(re, nullptr, PCRE_INFO_CAPTURECOUNT, &captureCount) != 0) {
        std::cerr << "Failed to get capture count." << std::endl;
        return false;
    }

    // ovector needs to be large enough to hold results
    // for three times the number of capturing groups plus one for the full match
    int ovectorSize = (captureCount + 1) * 3;
    int* ovector = new int[ovectorSize];

    int rc = pcre_exec(re, nullptr, subject, subjectLength, 0, 0, ovector, ovectorSize);
    if (rc >= 0) {
        std::cout << "Match found!" << std::endl;
        delete[] ovector;  // Free dynamically allocated memory
        return true;
    } else if (rc == PCRE_ERROR_NOMATCH) {
        std::cout << "No match found." << std::endl;
    } else {
        std::cerr << "Regex matching error occurred." << std::endl;
        switch (rc) {
            case PCRE_ERROR_NULL:
                std::cerr << "A NULL pointer was passed." << std::endl;
                break;
            case PCRE_ERROR_BADOPTION:
                std::cerr << "A bad option was passed." << std::endl;
                break;
            // Handle other specific error codes as necessary
            default:
                std::cerr << "Unknown error code: " << rc << std::endl;
        }
    }
    
    delete[] ovector;  // Free dynamically allocated memory
    return false;
}