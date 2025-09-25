bool matchRegex(pcre* re, const char* subject) {
    int subjectLength = strlen(subject);
    int ovector[30];  // This array will store captured substring positions
    int rc = pcre_exec(re, NULL, subject, subjectLength, 0, 0, ovector, 30);
    if (rc >= 0) {
        std::cout << "Match found!" << std::endl;
        return true;
    } else if (rc == PCRE_ERROR_NOMATCH) {
        std::cout << "No match found." << std::endl;
    } else {
        std::cerr << "Regex matching error: " << rc << std::endl;
    }
    return false;
}
