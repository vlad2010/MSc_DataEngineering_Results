#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <stdexcept>
#include <limits>
#include <memory>
#include <cstddef>

class SecureLogger {
public:
    explicit SecureLogger(std::filesystem::path log_dir, std::string base_name)
        : m_log_dir(std::move(log_dir)),
          m_name(sanitize_name(base_name)),
          m_file_number(0),
          m_bytes_written(0) {
        init_secure_log_dir();
        m_os = std::make_unique<std::ofstream>();
        // Enable exceptions so we don't ignore failures
        m_os->exceptions(std::ios::failbit | std::ios::badbit);
    }

    // Roll the log file securely
    void roll_file() {
        std::lock_guard<std::mutex> lock(m_mtx);

        if (m_os && m_os->is_open()) {
            try {
                m_os->flush();
                m_os->close();
            } catch (...) {
                // Ensure we close and continue rolling; rethrow if your policy requires it
            }
        }

        m_bytes_written = 0;

        if (m_file_number == std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("Log file counter overflow");
        }
        std::uint64_t next_num = ++m_file_number;

        // Build a safe filename under the secure log directory
        std::string file_name = m_name + "." + std::to_string(next_num) + ".txt";

        // Ensure the composed path stays within the secure directory
        std::filesystem::path full = m_log_dir / file_name;

        // Note: weakly_canonical won’t resolve non-existent final path components,
        // but parent_path() must resolve to the canonical secure directory.
        std::error_code ec;
        auto canonical_dir = std::filesystem::weakly_canonical(m_log_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to canonicalize log directory");
        }
        auto canonical_parent = std::filesystem::weakly_canonical(full.parent_path(), ec);
        if (ec || canonical_parent != canonical_dir) {
            throw std::runtime_error("Path traversal detected for log file");
        }

        // Recreate the ofstream to clear error state and ensure fresh open
        m_os = std::make_unique<std::ofstream>();
        m_os->exceptions(std::ios::failbit | std::ios::badbit);

        // Open the file (truncate) and then set restrictive permissions
        try {
            m_os->open(full, std::ios::out | std::ios::trunc | std::ios::binary);
        } catch (const std::ios_base::failure& e) {
            throw std::runtime_error(std::string("Failed to open log file: ") + e.what());
        }

        // Restrict file permissions: owner read/write only (best effort).
        // On POSIX this maps to 0600; on Windows this may be emulated and not strongly enforce ACLs.
        using perms = std::filesystem::perms;
        std::filesystem::permissions(full,
                                     perms::owner_read | perms::owner_write,
                                     std::filesystem::perm_options::replace,
                                     ec);
        // Non-fatal if not supported; consider logging ec.message().

        // Optional: fsync directory entry creation on POSIX for durability (not shown).
    }

private:
    static std::string sanitize_name(const std::string& in) {
        // Allow only alnum, dash, underscore; replace others with underscore.
        // Limit length to mitigate pathological inputs.
        std::string out;
        out.reserve(64);
        for (char c : in) {
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '-' || c == '_') {
                out.push_back(c);
            } else {
                out.push_back('_');
            }
            if (out.size() >= 64) break;
        }
        if (out.empty()) out = "log";
        return out;
    }

    void init_secure_log_dir() {
        // Create and secure the log directory
        std::error_code ec;
        std::filesystem::create_directories(m_log_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to create log directory");
        }

        // Reject symlinked directories to avoid redirection attacks (CWE-59)
        auto st = std::filesystem::symlink_status(m_log_dir, ec);
        if (ec || std::filesystem::is_symlink(st) || !std::filesystem::is_directory(st)) {
            throw std::runtime_error("Log directory is invalid or a symlink");
        }

        // Canonicalize and store the directory path
        m_log_dir = std::filesystem::weakly_canonical(m_log_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to canonicalize log directory");
        }

        // Restrict directory permissions to owner-only (best effort)
        using perms = std::filesystem::perms;
        std::filesystem::permissions(m_log_dir,
                                     perms::owner_all,
                                     std::filesystem::perm_options::replace,
                                     ec);
        // Non-fatal if not supported; consider logging ec.message().
    }

private:
    std::filesystem::path m_log_dir;               // secured, canonical absolute path
    std::string m_name;                            // sanitized base name
    std::uint64_t m_file_number;                   // guarded by m_mtx
    std::unique_ptr<std::ofstream> m_os;           // stream with exceptions enabled
    std::mutex m_mtx;                              // serialize roll_file and counter updates
    std::size_t m_bytes_written;
};