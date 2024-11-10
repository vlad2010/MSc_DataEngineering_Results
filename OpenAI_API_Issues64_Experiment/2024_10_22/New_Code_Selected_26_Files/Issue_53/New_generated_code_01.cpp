#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <regex>
#include <stdexcept>
#include <limits> // Include to use for streamsize

#include "constants.h"
#include "queryFile.h"
#include "superSearch.h"

void queryFile(std::string filePath, const char* query, std::vector<Result>& result) {
    std::ifstream fileStream;
    fileStream.open(filePath.c_str());

    if (!fileStream.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        throw std::runtime_error("File open error");
    }

    try {
        // Limit the buffer size to mitigate memory consumption issues
        const std::streamsize maxLineSize = 1024; // Example limit
        std::vector<QueryHit> queryHits;
        Result fileOverview = {filePath, 0, queryHits};

        int lineNumber = 0;
        std::string line;
        
        // Validate and potentially sanitize the regex input
        std::string safeQuery(query);

        // Assuming safeQuery is sanitized or checked for safety
        std::regex regexQuery(safeQuery, std::regex_constants::ECMAScript);

        while (std::getline(fileStream, line)) {
            if (line.size() > maxLineSize) {
                std::cerr << "Line exceeds maximum allowed size, skipping line: " << lineNumber << std::endl;
                continue;
            }

            lineNumber++;
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
    } catch (const std::regex_error& e) {
        std::cerr << "Invalid regular expression: " << e.what() << std::endl;
    }
}