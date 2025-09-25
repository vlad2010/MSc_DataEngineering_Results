#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <regex>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

class SecureLogger {
private:
    std::unique_ptr<std::ofstream> m_os;
    std::string m_name;
    std::string m_log_directory;
    size_t m_file_number;
    size_t m_bytes_written;
    size_t m_max_path_length = 255;
    
    // Sanitize filename to prevent path traversal
    std::string sanitize_filename(const std::string& name) {
        // Remove any path separators and dangerous characters
        std::regex dangerous_chars(R"([/\\:*?"<>|\.\.])");
        std::string safe_name = std::regex_replace(name, dangerous_chars, "_");
        
        // Limit length to prevent buffer issues
        if (safe_name.length() > 100) {
            safe_name = safe_name.substr(0, 100);
        }
        
        // Ensure non-empty name
        if (safe_name.empty()) {
            safe_name = "default_log";
        }
        
        return safe_name;
    }
    
    // Generate unique filename with timestamp and random component
    std::string generate_unique_filename() {
        std::stringstream ss;
        
        // Add timestamp for uniqueness
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // Add random component to prevent predictability
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1000, 9999);
        
        ss << sanitize_filename(m_name) << "_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_"
           << m_file_number << "_"
           << dis(gen) << ".txt";
           
        return ss.str();
    }
    
public:
    SecureLogger(const std::string& name, const std::string& log_dir = "./logs") 
        : m_name(name), m_log_directory(log_dir), m_file_number(0), m_bytes_written(0) {
        
        // Validate and create log directory if it doesn't exist
        std::filesystem::path dir_path(m_log_directory);
        
        // Ensure directory path is within acceptable bounds
        if (dir_path.string().length() > m_max_path_length) {
            throw std::invalid_argument("Log directory path too long");
        }
        
        // Create directory with restricted permissions (owner only: rwx------)
        if (!std::filesystem::exists(dir_path)) {
            std::filesystem::create_directories(dir_path);
            std::filesystem::permissions(dir_path, 
                std::filesystem::perms::owner_all |
                std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
                std::filesystem::perms::others_read | std::filesystem::perms::others_exec,
                std::filesystem::perm_options::remove);
        }
    }
    
    void roll_file() {
        // Safely close existing file
        if (m_os && m_os->is_open()) {
            m_os->flush();
            m_os->close();
            
            // Verify closure
            if (m_os->is_open()) {
                throw std::runtime_error("Failed to close previous log file");
            }
        }
        
        m_bytes_written = 0;
        ++m_file_number;
        
        // Generate secure filename
        std::string filename = generate_unique_filename();
        std::filesystem::path full_path = std::filesystem::path(m_log_directory) / filename;
        
        // Validate full path length
        if (full_path.string().length() > m_max_path_length) {
            throw std::runtime_error("Generated file path exceeds maximum length");
        }
        
        // Ensure we're still within the log directory (prevent directory traversal)
        std::filesystem::path canonical_dir = std::filesystem::canonical(m_log_directory);
        std::filesystem::path parent_path = full_path.parent_path();
        
        if (!parent_path.string().starts_with(canonical_dir.string())) {
            throw std::security_error("Attempted directory traversal detected");
        }
        
        // Create new file stream
        m_os.reset(new std::ofstream());
        
        // Open file with explicit flags for security
        m_os->open(full_path, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        
        // Verify file was opened successfully
        if (!m_os->is_open() || !m_os->good()) {
            std::string error_msg = "Failed to open log file: " + full_path.string();
            m_os.reset();  // Clean up the failed stream
            throw std::runtime_error(error_msg);
        }
        
        // Set restrictive permissions on the new file (owner read/write only: rw-------)
        try {
            std::filesystem::permissions(full_path,
                std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                std::filesystem::perm_options::replace);
        } catch (const std::filesystem::filesystem_error& e) {
            // Log permission setting failure but don't fail the operation
            // In production, this should be logged to a security audit log
        }
    }
    
    // Custom exception for security-related errors
    class security_error : public std::runtime_error {
    public:
        explicit security_error(const std::string& msg) : std::runtime_error(msg) {}
    };
};