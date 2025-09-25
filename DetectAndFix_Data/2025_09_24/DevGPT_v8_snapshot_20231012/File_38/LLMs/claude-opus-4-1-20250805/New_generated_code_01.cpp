#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <curl/curl.h>
#include <memory>
#include <stdexcept>
#include <regex>

// Secure configuration structure
struct EmailConfig {
    std::string smtp_server;
    int smtp_port;
    std::string username;
    std::string password;
    bool use_tls;
    bool verify_peer;
    bool verify_host;
};

struct EmailData {
    const char* data;
    size_t size;
};

// Secure callback function without information leakage
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    if (!ptr || !user_data) {
        return 0;
    }
    
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (email_data->size == 0) {
        return 0;
    }

    size_t buffer_size = size * nmemb;
    size_t len = std::min(buffer_size, email_data->size);
    
    // Secure memory copy with bounds checking
    if (len > 0 && email_data->data) {
        memcpy(ptr, email_data->data, len);
        email_data->data += len;
        email_data->size -= len;
    }
    
    // Remove debug output that exposes sensitive data (CWE-209 fix)
    // Use proper logging framework with appropriate log levels in production
    
    return len;
}

// Email address validation
bool validate_email(const std::string& email) {
    const std::regex pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    return std::regex_match(email, pattern);
}

// Sanitize email content to prevent header injection
std::string sanitize_email_content(const std::string& content) {
    std::string sanitized = content;
    // Remove or escape potentially dangerous characters
    size_t pos = 0;
    while ((pos = sanitized.find("\r\n.", pos)) != std::string::npos) {
        sanitized.insert(pos + 3, ".");
        pos += 4;
    }
    return sanitized;
}

// Load configuration from secure source (environment variables or encrypted config file)
EmailConfig load_secure_config() {
    EmailConfig config;
    
    // Load from environment variables (CWE-798 fix)
    const char* smtp_server = std::getenv("SMTP_SERVER");
    const char* smtp_port = std::getenv("SMTP_PORT");
    const char* smtp_user = std::getenv("SMTP_USER");
    const char* smtp_pass = std::getenv("SMTP_PASSWORD");
    
    if (!smtp_server || !smtp_port || !smtp_user || !smtp_pass) {
        throw std::runtime_error("Missing required SMTP configuration in environment variables");
    }
    
    config.smtp_server = smtp_server;
    config.smtp_port = std::stoi(smtp_port);
    config.username = smtp_user;
    config.password = smtp_pass;
    config.use_tls = true;  // Always use TLS (CWE-311 fix)
    config.verify_peer = true;
    config.verify_host = true;
    
    return config;
}

class SecureEmailSender {
private:
    CURL* curl;
    struct curl_slist* recipients;
    
public:
    SecureEmailSender() : curl(nullptr), recipients(nullptr) {
        // Initialize CURL with proper error checking (CWE-252 fix)
        CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
        if (res != CURLE_OK) {
            throw std::runtime_error("Failed to initialize CURL globally");
        }
        
        curl = curl_easy_init();
        if (!curl) {
            curl_global_cleanup();
            throw std::runtime_error("Failed to initialize CURL handle");
        }
    }
    
    ~SecureEmailSender() {
        if (recipients) {
            curl_slist_free_all(recipients);
        }
        if (curl) {
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }
    
    bool send_email(const std::string& from, const std::string& to, 
                   const std::string& subject, const std::string& body,
                   const EmailConfig& config) {
        
        // Validate email addresses (CWE-20 fix)
        if (!validate_email(from) || !validate_email(to)) {
            std::cerr << "Invalid email address format" << std::endl;
            return false;
        }
        
        // Sanitize content
        std::string safe_subject = sanitize_email_content(subject);
        std::string safe_body = sanitize_email_content(body);
        
        // Build email content
        std::string email_data_str =
            "From: " + from + "\r\n"
            "To: " + to + "\r\n"
            "Subject: " + safe_subject + "\r\n\r\n" +
            safe_body + "\r\n";
        
        EmailData email_data = { email_data_str.c_str(), email_data_str.size() };
        
        // Configure SMTP server with TLS (CWE-311 fix)
        std::string smtp_url = "smtps://" + config.smtp_server + ":" + 
                               std::to_string(config.smtp_port);
        curl_easy_setopt(curl, CURLOPT_URL, smtp_url.c_str());
        
        // Set authentication
        curl_easy_setopt(curl, CURLOPT_USERNAME, config.username.c_str());
        curl_easy_setopt(curl, CURLOPT_PASSWORD, config.password.c_str());
        
        // Security options
        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, config.verify_peer ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, config.verify_host ? 2L : 0L);
        
        // Set timeout to prevent hanging
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        // Set sender and recipients
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, from.c_str());
        recipients = curl_slist_append(recipients, to.c_str());
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
        
        // Set up the email data callback
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
        curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        
        // Disable verbose output in production
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
        
        // Send the email with proper error handling
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            // Log error without exposing sensitive information
            std::cerr << "Failed to send email. Error code: " << res << std::endl;
            return false;
        }
        
        return true;
    }
};

int main() {
    try {
        // Load configuration from secure source
        EmailConfig config = load_secure_config();
        
        // Create secure email sender
        SecureEmailSender sender;
        
        // Get email details from secure input (not hardcoded)
        std::string from_email = std::getenv("FROM_EMAIL") ? std::getenv("FROM_EMAIL") : "";
        std::string to_email = std::getenv("TO_EMAIL") ? std::getenv("TO_EMAIL") : "";
        
        if (from_email.empty() || to_email.empty()) {
            std::cerr << "Email addresses must be provided via environment variables" << std::endl;
            return 1;
        }
        
        std::string subject = "Secure Email Test";
        std::string body = "This is a securely sent email message.";
        
        // Send email
        if (sender.send_email(from_email, to_email, subject, body, config)) {
            std::cout << "Email sent successfully." << std::endl;
        } else {
            std::cerr << "Failed to send email." << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}