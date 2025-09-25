#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <cctype>
#include <algorithm>

// Helper function to sanitize file names (basic example)
std::string sanitize_filename(const std::string& name) {
    std::string sanitized;
    for (char c : name) {
        // Allow only alphanumeric, dash, underscore, and dot
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.') {
            sanitized += c;
        } else {
            sanitized += '_'; // Replace unsafe chars
        }
    }
    return sanitized;
}

void roll_file() {
    try {
        if (m_os && m_os->is_open()) {
            m_os->flush();
            m_os->close();
        }
    } catch (const std::exception& e) {
        // Log error or handle as appropriate
        // For now, rethrow to avoid silent failure
        throw;
    }

    m_bytes_written = 0;
    m_os.reset(new std::ofstream());

    // Sanitize m_name to prevent path traversal
    std::string log_file_name = sanitize_filename(m_name);
    log_file_name.append(".");
    log_file_name.append(std::to_string(++m_file_number));
    log_file_name.append(".txt");

    m_os->open(log_file_name, std::ofstream::out | std::ofstream::trunc);

    // Check if file was opened successfully
    if (!m_os->is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_file_name);
    }
}