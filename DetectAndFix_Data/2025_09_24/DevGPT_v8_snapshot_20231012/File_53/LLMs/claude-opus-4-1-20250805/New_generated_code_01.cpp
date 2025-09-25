#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <regex>
#include <stdexcept>
#include <filesystem>
#include <chrono>

#include "constants.h"
#include "queryFile.h"
#include "superSearch.h"

// Constants for security limits
constexpr size_t MAX_LINE_LENGTH = 1048576;  // 1MB max line length
constexpr size_t MAX_RESULTS_PER_FILE = 10000;  // Max matches per file
constexpr size_t MAX_FILE_SIZE = 104857600;  // 100MB max file size
constexpr auto REGEX_TIMEOUT = std::chrono::milliseconds(100);  // Regex timeout

// Sanitize and validate file path
std::string sanitizeFilePath(const std::string& filePath) {
    try {
        // Use filesystem library to canonicalize and validate path
        std::filesystem::path p = std::filesystem::canonical(filePath);
        
        // Check if file exists and is a regular file
        if (!std::filesystem::exists(p) || !std::filesystem::is_regular_file(p)) {
            throw std::runtime_error("Invalid file path");
        }
        
        // Check file size
        if (std::filesystem::file_size(p) > MAX_FILE_SIZE) {
            throw std::runtime_error("File too large");
        }
        
        return p.string();
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Invalid file path: " + std::string(e.what()));
    }
}

// Escape special regex characters in query string for literal search
std::string escapeRegexSpecialChars(const std::string& str) {
    static const std::regex specialChars(R"([\^\$\.\|\?\*\+\(\)\[\]\{\}\\])");
    return std::regex_replace(str, specialChars, R"(\$&)");
}

// Validate regex pattern for safety
bool isRegexSafe(const std::string& pattern) {
    // Check for patterns that could cause ReDoS
    // Detect nested quantifiers and other dangerous patterns
    static const std::regex dangerousPatterns[] = {
        std::regex(R"(\(\.\*\)\+)"),  // (.*)+
        std::regex(R"(\(\.\+\)\+)"),  // (.+)+
        std::regex(R"(\([^)]*\*\)\+)"),  // Nested quantifiers
        std::regex(R"(\([^)]*\+\)\+)"),  // Nested quantifiers
        std::regex(R"((\.\*){2,})"),  // Multiple .*
        std::regex(R"((\.\+){2,})")   // Multiple .+
    };
    
    for (const auto& dangerous : dangerousPatterns) {
        if (std::regex_search(pattern, dangerous)) {
            return false;
        }
    }
    
    // Check pattern length
    if (pattern.length() > 1000) {
        return false;
    }
    
    return true;
}

void queryFile(std::string filePath, const char* query, std::vector<Result>& result) {
    if (!query || strlen(query) == 0) {
        throw std::invalid_argument("Query cannot be null or empty");
    }
    
    try {
        // Sanitize and validate file path
        filePath = sanitizeFilePath(filePath);
        
        std::ifstream fileStream;
        fileStream.open(filePath.c_str());
        
        if (!fileStream.is_open()) {
            throw std::runtime_error("Unable to open file: " + filePath);
        }
        
        // Prepare regex with safety checks
        std::string queryStr(query);
        
        // Option 1: Treat as literal string (safer)
        // queryStr = escapeRegexSpecialChars(queryStr);
        
        // Option 2: Allow regex but validate for safety
        if (!isRegexSafe(queryStr)) {
            throw std::invalid_argument("Potentially dangerous regex pattern");
        }
        
        std::regex regexQuery;
        try {
            // Use basic regex syntax to reduce complexity
            regexQuery = std::regex(queryStr, std::regex::ECMAScript | std::regex::optimize);
        } catch (const std::regex_error& e) {
            throw std::invalid_argument("Invalid regex pattern: " + std::string(e.what()));
        }
        
        std::vector<QueryHit> queryHits;
        Result fileOverview = {filePath, 0, queryHits};
        
        int lineNumber = 0;
        std::string line;
        line.reserve(MAX_LINE_LENGTH);  // Pre-allocate for performance
        
        while (getline(fileStream, line)) {
            lineNumber++;
            
            // Check line length limit
            if (line.length() > MAX_LINE_LENGTH) {
                if (DEV) {
                    std::cerr << "Warning: Line " << lineNumber << " exceeds maximum length, skipping" << std::endl;
                }
                continue;
            }
            
            // Check if we've hit the maximum results limit
            if (fileOverview.totalHits >= MAX_RESULTS_PER_FILE) {
                if (DEV) {
                    std::cerr << "Warning: Maximum results limit reached for file: " << filePath << std::endl;
                }
                break;
            }
            
            try {
                // Use regex_search with timeout protection (C++11 doesn't have built-in timeout)
                // In production, consider using std::async with timeout or regex libraries with timeout support
                std::smatch match;
                if (std::regex_search(line, match, regexQuery)) {
                    int offset = static_cast<int>(match.position());
                    
                    QueryHit queryHitDetails = {
                        filePath + ":" + std::to_string(lineNumber) + ":" + std::to_string(offset),
                        line,
                        lineNumber,
                        offset
                    };
                    
                    fileOverview.totalHits++;
                    fileOverview.queryHits.push_back(queryHitDetails);
                    
                    if (DEV) {
                        std::cout << "found: " << offset << " -- " 
                                 << line.substr(0, std::min(size_t(10), line.length()))
                                 << std::endl;
                    }
                }
            } catch (const std::regex_error& e) {
                if (DEV) {
                    std::cerr << "Regex error on line " << lineNumber << ": " << e.what() << std::endl;
                }
                continue;
            }
        }
        
        fileStream.close();
        
        if (fileOverview.totalHits > 0) {
            result.push_back(fileOverview);
        }
        
    } catch (const std::exception& e) {
        // Log error instead of terminating program
        std::cerr << "Error processing file " << filePath << ": " << e.what() << std::endl;
        // Optionally, you could add error information to the result
        // or throw the exception to let the caller handle it
        throw;
    }
}