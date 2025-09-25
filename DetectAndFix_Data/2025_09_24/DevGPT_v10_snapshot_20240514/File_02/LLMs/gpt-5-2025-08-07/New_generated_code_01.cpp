#include <cstdint>
#include <ctime>
#include <cstdio>
#include <limits>
#include <ostream>
#include <stdexcept>

void format_timestamp(std::ostream& os, uint64_t timestamp_us) {
    // Split into seconds and microseconds
    const uint64_t seconds64 = timestamp_us / 1000000ULL;
    const uint32_t micros = static_cast<uint32_t>(timestamp_us % 1000000ULL);

    // Validate range before narrowing to time_t (avoid CWE-197/CWE-681)
    if (seconds64 > static_cast<uint64_t>(std::numeric_limits<std::time_t>::max())) {
        throw std::overflow_error("timestamp out of range for time_t");
    }

    std::time_t tt = static_cast<std::time_t>(seconds64);

    // Thread-safe UTC conversion (avoid CWE-676/CWE-362)
    std::tm tm_utc{};
#if defined(_WIN32)
    errno_t err = gmtime_s(&tm_utc, &tt);
    if (err != 0) {
        throw std::runtime_error("gmtime_s failed");
    }
#else
    if (gmtime_r(&tt, &tm_utc) == nullptr) {
        throw std::runtime_error("gmtime_r failed");
    }
#endif

    // Format date/time safely and check result (avoid CWE-252/CWE-457)
    char datebuf[32]; // sufficient for "%Y-%m-%d %H:%M:%S." (21 chars + NUL)
    if (std::strftime(datebuf, sizeof(datebuf), "%Y-%m-%d %H:%M:%S.", &tm_utc) == 0) {
        // 0 means buffer too small or error
        throw std::runtime_error("strftime failed or buffer too small");
    }

    // Format microseconds in a bounded way (avoid CWE-120/CWE-242/CWE-676)
    char microbuf[7]; // 6 digits + NUL
    int n = std::snprintf(microbuf, sizeof(microbuf), "%06u", micros);
    if (n != 6) {
        throw std::runtime_error("failed to format microseconds");
    }

    os << '[' << datebuf << microbuf << ']';
}