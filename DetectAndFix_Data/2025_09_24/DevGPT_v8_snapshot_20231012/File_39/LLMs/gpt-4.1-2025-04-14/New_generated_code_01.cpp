#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <curl/curl.h>
#include <sys/stat.h>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Securely zero memory for sensitive data
void secure_zero(std::string& s) {
    volatile char* p = const_cast<volatile char*>(s.data());
    std::fill(p, p + s.size(), 0);
}

// Check file permissions: only owner can read/write
bool check_file_permissions(const std::string& filename) {
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) return false;
    // Only owner can read/write (0600)
    return (st.st_mode & 0777) == 0600;
}

// Basic email validation (very simple, for demo)
bool is_valid_email(const std::string& email) {
    auto at = email.find('@');
    auto dot = email.find('.', at);
    return at != std::string::npos && dot != std::string::npos && at > 0 && dot > at + 1;
}

// Sanitize header values (remove CR/LF)
std::string sanitize_header(const std::string& input) {
    std::string out;
    for (char c : input) {
        if (c != '\r' && c != '\n') out += c;
    }
    return out;
}

struct EmailData {
    const char* data;
    size_t total_size;
    size_t bytes_read;
};

// Callback function to read data from the email body
static size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *userp) {
    EmailData *email_data = static_cast<EmailData *>(userp);
    size_t room = size * nmemb;
    if (email_data->bytes_read >= email_data->total_size) return 0;
    size_t len = std::min(room, email_data->total_size - email_data->bytes_read);
    memcpy(ptr, email_data->data + email_data->bytes_read, len);
    email_data->bytes_read += len;
    return len;
}

bool get_password(std::string &password) {
    const std::string secret_file_path = "./secret";
    if (!check_file_permissions(secret_file_path)) {
        std::cerr << "Secret file permissions are not secure (should be 0600)" << std::endl;
        return false;
    }
    std::ifstream secret_file(secret_file_path);
    if (secret_file.is_open()) {
        getline(secret_file, password);
        secret_file.close();
        return true;
    }
    return false;
}

class EmailSender {
public:
    EmailSender() {
        curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Error initializing libcurl." << std::endl;
            return;
        }
        // Remove verbose output in production
        // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        const std::string url = "smtps://smtp.gmail.com:465";
        const std::string username = "watchtower.test2023@gmail.com";

        if (!get_password(password)) {
            std::cerr << "Error obtaining the password" << std::endl;
            return;
        }

        const std::string mail_from = "watchtower.test2023@gmail.com";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
        curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from.c_str());
        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
    }

    ~EmailSender() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
        secure_zero(password);
    }

    bool sendEmail(const std::vector<std::string>& recipients, const std::string& subject, const std::string& body) {
        // Validate and sanitize recipients
        std::vector<std::string> safe_recipients;
        for (const auto& r : recipients) {
            if (!is_valid_email(r)) {
                std::cerr << "Invalid recipient email: " << r << std::endl;
                return false;
            }
            safe_recipients.push_back(sanitize_header(r));
        }

        // Sanitize subject and body
        std::string safe_subject = sanitize_header(subject);

        std::string email_content = "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n";
        email_content += "To: ";
        for (size_t i = 0; i < safe_recipients.size(); ++i) {
            email_content += safe_recipients[i];
            if (i < safe_recipients.size() - 1) {
                email_content += ",";
            }
        }
        email_content += "\r\n";
        email_content += "Subject: " + safe_subject + "\r\n";
        email_content += "\r\n";
        email_content += body;

        EmailData email_data = { email_content.c_str(), email_content.size(), 0 };
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
        curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

        struct curl_slist *recipients_list = NULL;
        for (const std::string& recipient : safe_recipients) {
            recipients_list = curl_slist_append(recipients_list, recipient.c_str());
        }
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients_list);

        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(recipients_list);

        if (res != CURLE_OK) {
            std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
        return true;
    }

private:
    CURL *curl = nullptr;
    std::string password;
};

int main() {
    EmailSender emailSender;

    std::vector<std::string> recipients = {"recipient1@example.com", "recipient2@example.com"};
    std::string subject = "Test Subject";
    std::string body = "This is the body of the email.";

    if (emailSender.sendEmail(recipients, subject, body)) {
        std::cout << "Email sent successfully." << std::endl;
    } else {
        std::cerr << "Failed to send the email." << std::endl;
    }

    return 0;
}