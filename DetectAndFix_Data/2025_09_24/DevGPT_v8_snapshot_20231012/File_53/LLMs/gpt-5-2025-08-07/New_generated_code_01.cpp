#include <fstream>
#include <iostream>
#include <vector>
#include <regex>
#include <string>
#include <limits>
#include <filesystem>   // C++17
#include <system_error>

#include "constants.h"
#include "queryFile.h"
#include "superSearch.h"

// Assuming these exist elsewhere:
// struct QueryHit { std::string location; std::string line; int lineNumber; int offset; };
// struct Result { std::string filePath; size_t totalHits; std::vector<QueryHit> queryHits; };

// Escape regex metacharacters for literal searches
static std::string escapeRegex(const std::string& s) {
    static const std::regex metachars{R"([.^$|()\\[\]{}*+?])"};
    return std::regex_replace(s, metachars, R"(\$&)");
}

// Simple, optional path validation. In a real system, enforce an allowed root.
static bool isPathSane(const std::string& filePath) {
    if (filePath.find('\0') != std::string::npos) return false; // Embedded NUL
    try {
        std::error_code ec;
        auto p = std::filesystem::path(filePath);
        // Optional: reject directories; we only read regular files
        auto status = std::filesystem::status(p, ec);
        if (ec) return false;
        if (!std::filesystem::is_regular_file(status)) return false;
    } catch (...) {
        return false;
    }
    return true;
}

void queryFile(std::string filePath, const char* query, std::vector<Result>& result) {
    // Input validation
    if (query == nullptr) {
        std::cerr << "Invalid query (null)\n";
        return; // CWE-476 fix
    }
    const std::string queryStr(query);

    // Defend against absurdly long patterns (helps with CWE-1333)
    static constexpr size_t kMaxQueryLength = 256; // tune as appropriate
    if (queryStr.size() > kMaxQueryLength) {
        std::cerr << "Query too long\n";
        return;
    }

    // Optional path checks (mitigate CWE-22 in contexts with untrusted input)
    if (!isPathSane(filePath)) {
        std::cerr << "Invalid or unsupported file path\n";
        return;
    }

    // Open file with RAII, do not exit() on error (CWE-703)
    std::ifstream fileStream(filePath, std::ios::in);
    if (!fileStream.is_open()) {
        std::cerr << "Unable to open file\n"; // avoid leaking full path (CWE-200/209)
        return;
    }

    // Build regex safely:
    // If your intent is literal substring search by default, escape metacharacters.
    // If you truly need regex semantics from untrusted input, strongly consider a safe engine (e.g., RE2).
    bool treatInputAsRegex = false; // set true only if input is trusted/validated
    std::string pattern = treatInputAsRegex ? queryStr : escapeRegex(queryStr);

    std::regex regexQuery;
    try {
        regexQuery = std::regex(pattern, std::regex_constants::ECMAScript | std::regex_constants::optimize);
    } catch (const std::regex_error&) {
        std::cerr << "Invalid search pattern\n"; // CWE-703 fix
        return;
    }

    std::vector<QueryHit> queryHits;
    Result fileOverview = {filePath, 0u, queryHits};

    size_t lineNumber = 0;
    std::string line;

    // Resource guards
    static constexpr size_t kMaxLineLength = 1 << 20; // 1 MiB per line cap
    static constexpr size_t kMaxHitsPerFile = 100000; // cap to prevent memory blowup

    while (std::getline(fileStream, line)) {
        ++lineNumber;

        // Optionally skip processing of extremely long lines to avoid DoS (CWE-400)
        if (line.size() > kMaxLineLength) {
            // Either skip, or process the first chunk only. Here we skip to be safe.
            continue;
        }

        // Find first match in the line; optionally iterate for multiple matches
        std::smatch match;
        std::string::const_iterator searchStart(line.cbegin());
        bool matched = std::regex_search(searchStart, line.cend(), match, regexQuery);

        if (matched) {
            // Compute offset safely
            std::ptrdiff_t pos = match.position();
            if (pos < 0) pos = 0; // defensive
            size_t offset = static_cast<size_t>(pos);

            // Guard conversions to int for external structs
            int safeLineNumber = (lineNumber > static_cast<size_t>(std::numeric_limits<int>::max()))
                                     ? std::numeric_limits<int>::max()
                                     : static_cast<int>(lineNumber);
            int safeOffset = (offset > static_cast<size_t>(std::numeric_limits<int>::max()))
                                 ? std::numeric_limits<int>::max()
                                 : static_cast<int>(offset);

            QueryHit queryHitDetails = {
                filePath + ":" + std::to_string(lineNumber) + ":" + std::to_string(offset),
                line,
                safeLineNumber,
                safeOffset};

            ++fileOverview.totalHits;
            fileOverview.queryHits.push_back(std::move(queryHitDetails));

            if (fileOverview.totalHits >= kMaxHitsPerFile) {
                std::cerr << "Hit limit reached for file, stopping early\n";
                break;
            }
        }
    }

    // fileStream closed automatically via RAII

    if (fileOverview.totalHits > 0) {
        result.push_back(std::move(fileOverview));
    }
}