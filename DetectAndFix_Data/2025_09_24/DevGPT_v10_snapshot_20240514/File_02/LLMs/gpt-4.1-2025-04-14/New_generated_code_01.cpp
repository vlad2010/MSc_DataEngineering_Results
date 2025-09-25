#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <cstdint>
#include <cstdio>
#include <cstring>

void format_timestamp(std::ostream& os, uint64_t timestamp) {
    std::time_t time_t_val = timestamp / 1000000;
    struct tm tm_buf;
    // Use gmtime_r for thread safety (POSIX). On Windows, use gmtime_s.
    struct tm* gmtime_res = gmtime_r(&time_t_val, &tm_buf);
    if (!gmtime_res) {
        os << "[invalid time]";
        return;
    }

    char buffer[32];
    // Correct format string and check return value
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %T.", gmtime_res) == 0) {
        os << "[invalid time]";
        return;
    }

    char microseconds[7];
    // Use snprintf for safety, and correct format specifier for uint64_t
    if (std::snprintf(microseconds, sizeof(microseconds), "%06" PRIu64, timestamp % 1000000) < 0) {
        os << "[invalid time]";
        return;
    }

    os << '[' << buffer << microseconds << ']';
}