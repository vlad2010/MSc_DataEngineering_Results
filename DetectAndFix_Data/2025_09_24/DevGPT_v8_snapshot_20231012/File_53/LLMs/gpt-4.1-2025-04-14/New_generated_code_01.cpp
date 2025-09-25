#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <regex>
#include <stdexcept>
#include <limits>
#include <filesystem>

#include "constants.h"
#include "queryFile.h"
#include "superSearch.h"

// Helper function to validate file path (CWE-22)
bool isValidFilePath(const std::string& filePath, const std::string& allowedDir) {
    try {
        std::filesystem::path absPath = std::filesystem::absolute(filePath);
        std::filesystem::path absAllowedDir = std::filesystem::absolute(allowedDir);
        return std::mismatch(absAllowedDir.begin(), absAllowedDir.end(), absPath.begin()).first == absAllowedDir.end();
    } catch (...) {
        return false;
    }
}

void queryFile(std::string filePath, const char* query, std::vector<Result>& result) {
    // Restrict file access to a specific directory (e.g., "./data")
    const std::string allowedDirectory = "./data";
    if (!isValidFilePath(filePath, allowedDirectory)) {
        std::cerr << "Access to the specified file is not allowed." << std::endl;
        return;
    }

    std::ifstream fileStream(filePath.c_str(), std::ios::in);
    if (!fileStream.is_open()) {
        std::cerr << "Unable to open file." << std::endl; // Avoid leaking file path
        return;
    }

    std::vector<QueryHit> queryHits;
    Result fileOverview = {filePath, 0, queryHits};

    int lineNumber = 0;
    std::string line;

    // Validate and construct regex safely (CWE-625, CWE-248)
    std::regex regexQuery;
    try {
        regexQuery = std::regex(query);
    } catch (const std::regex_error& e) {
        std::cerr << "Invalid regular expression." << std::endl;
        fileStream.close();
        return;
    }

    // Limit the number of lines processed (CWE-400)
    const int MAX_LINES = 100000; // Example limit

    while (getline(fileStream, line)) {
        lineNumber++;
        if (lineNumber > MAX_LINES) {
            std::cerr << "File too large, processing stopped." << std::endl;
            break;
        }
        std::smatch match;
        if (std::regex_search(line, match, regexQuery)) {
            int offset = match.position();

            QueryHit queryHitDetails = {filePath + ":" + std::to_string(lineNumber) + ":" + std::to_string(offset),
                                        line,
                                        lineNumber,
                                        offset};
            fileOverview.totalHits++;
            fileOverview.queryHits.push_back(queryHitDetails);

            if (DEV)
                std::cout << "found: " << offset << " -- " << line.substr(0, 10)
                          << std::endl;
        }
    }

    fileStream.close();
    if (fileOverview.totalHits > 0) {
        result.push_back(fileOverview);
    }
}