#include <iostream>
#include <cstring>
#include <pcre.h>

bool matchRegex(pcre* re, const char* subject) {
    // Check for NULL pointers (CWE-476)
    if (re == nullptr) {
        std::cerr << "Regex object is NULL." << std::endl;
        return false;
    }
    if (subject == nullptr) {
        std::cerr << "Subject string is NULL." << std::endl;
        return false;
    }

    // Defensive: limit subject length to a reasonable value (optional)
    size_t subjectLength = strlen(subject);
    if (subjectLength > 1000000) { // Arbitrary large limit
        std::cerr << "Subject string too long." << std::endl;
        return false;
    }

    int ovector[30];  // This array will store captured substring positions
    int rc = pcre_exec(re, NULL, subject, static_cast<int>(subjectLength), 0, 0, ovector, 30);
    if (rc >= 0) {
        std::cout << "Match found!" << std::endl;
        return true;
    } else if (rc == PCRE_ERROR_NOMATCH) {
        std::cout << "No match found." << std::endl;
    } else {
        // Avoid leaking internal error codes (CWE-209)
        std::cerr << "Regex matching error occurred." << std::endl;
        // Optionally, log rc to a secure log for debugging
    }
    return false;
}