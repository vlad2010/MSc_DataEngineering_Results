#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <regex>
#include <limits>
#include <stdexcept>

// Security-hardened file reading function
bool secureFileRead(int& cases) {
    const std::string ALLOWED_DIRECTORY = "./data/";  // Define allowed directory
    const size_t MAX_FILENAME_LENGTH = 255;
    const std::regex VALID_FILENAME_PATTERN("^[a-zA-Z0-9_\\-]+\\.(txt|dat)$");
    
    std::string filename;
    
    std::cout << "Enter filename: ";
    
    // Use getline instead of >> to prevent buffer overflow
    // and to handle spaces in filenames properly
    std::getline(std::cin, filename);
    
    // Input validation: Check filename length
    if (filename.empty() || filename.length() > MAX_FILENAME_LENGTH) {
        std::cerr << "Error: Invalid filename length" << std::endl;
        return false;
    }
    
    // Input validation: Check for null bytes
    if (filename.find('\0') != std::string::npos) {
        std::cerr << "Error: Invalid characters in filename" << std::endl;
        return false;
    }
    
    // Input validation: Validate filename pattern (alphanumeric, underscore, hyphen only)
    if (!std::regex_match(filename, VALID_FILENAME_PATTERN)) {
        std::cerr << "Error: Invalid filename format. Only alphanumeric characters, "
                  << "underscores, and hyphens allowed with .txt or .dat extension" << std::endl;
        return false;
    }
    
    // Construct safe path within allowed directory
    std::filesystem::path safePath = std::filesystem::path(ALLOWED_DIRECTORY) / filename;
    
    // Canonicalize the path to resolve any symbolic links and normalize it
    std::error_code ec;
    std::filesystem::path canonicalPath = std::filesystem::canonical(safePath, ec);
    
    // If file doesn't exist, use weakly_canonical instead
    if (ec) {
        canonicalPath = std::filesystem::weakly_canonical(safePath);
    }
    
    // Verify the resolved path is within the allowed directory
    std::filesystem::path allowedCanonical = std::filesystem::canonical(ALLOWED_DIRECTORY, ec);
    if (ec) {
        allowedCanonical = std::filesystem::weakly_canonical(ALLOWED_DIRECTORY);
    }
    
    // Check if the file path starts with the allowed directory path
    std::string canonicalStr = canonicalPath.string();
    std::string allowedStr = allowedCanonical.string();
    
    if (canonicalStr.find(allowedStr) != 0) {
        std::cerr << "Error: Access denied - file outside allowed directory" << std::endl;
        return false;
    }
    
    // Check file permissions and existence
    if (!std::filesystem::exists(canonicalPath)) {
        std::cerr << "Error: File does not exist" << std::endl;
        return false;
    }
    
    if (!std::filesystem::is_regular_file(canonicalPath)) {
        std::cerr << "Error: Not a regular file" << std::endl;
        return false;
    }
    
    // Open file with explicit mode
    std::ifstream infile(canonicalPath, std::ios::in);
    
    // Check if file opened successfully
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file" << std::endl;
        return false;
    }
    
    // Check if file is readable
    if (!infile.good()) {
        std::cerr << "Error: File is not readable" << std::endl;
        infile.close();
        return false;
    }
    
    // Read the integer value with error checking
    if (!(infile >> cases)) {
        std::cerr << "Error: Failed to read integer from file" << std::endl;
        infile.close();
        return false;
    }
    
    // Validate the read value is within reasonable bounds
    const int MAX_CASES = 1000000;  // Define reasonable maximum
    const int MIN_CASES = 0;
    
    if (cases < MIN_CASES || cases > MAX_CASES) {
        std::cerr << "Error: Invalid number of cases (out of range)" << std::endl;
        infile.close();
        return false;
    }
    
    // Clean up
    infile.close();
    
    return true;
}

// Usage example
int main() {
    int cases = 0;
    
    if (secureFileRead(cases)) {
        std::cout << "Successfully read " << cases << " cases from file" << std::endl;
        // Process cases...
    } else {
        std::cerr << "Failed to read file securely" << std::endl;
        return 1;
    }
    
    return 0;
}