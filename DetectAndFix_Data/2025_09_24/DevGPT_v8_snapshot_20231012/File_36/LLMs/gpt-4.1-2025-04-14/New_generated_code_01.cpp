#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>
#include <cstring>
#include <curl/curl.h>

// Securely zero memory (portable)
void secure_zero(void* v, size_t n) {
    volatile char* p = (volatile char*)v;
    while (n--) *p++ = 0;
}

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *userp) {
    std::string* data = static_cast<std::string*>(userp);
    if (data->empty()) {
        return 0;
    }
    size_t len = std::min(size * nmemb, data->size());
    memcpy(ptr, data->data(), len);
    data->erase(0, len);
    return len;
}

// Check file permissions (POSIX only)
bool is_file_secure(const std::string& filename) {
#ifdef __unix__
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) return false;
    // Only owner can read/write
    if ((st.st_mode & 077) != 0) return false;
#endif
    return true;
}

int main(int argc, char* argv[]) {
    // Get credentials from environment variables
    const char* env_username = std::getenv("EMAIL_USERNAME");
    const char* env_password_file = std::getenv("EMAIL_PASSWORD_FILE");
    const char* env_mail_from = std::getenv("EMAIL_FROM");
    const char* env_mail_rcpt = std::getenv("EMAIL_TO");

    if (!env_username || !env_password_file || !env_mail_from || !env_mail_rcpt) {
        std::cerr << "Missing environment variables for credentials." << std::endl;
        return 1;
    }

    std::string username(env_username);
    std::string password_file(env_password_file);
    std::string mail_from(env_mail_from);
    std::string mail_rcpt(env_mail_rcpt);

    // Check file permissions
    if (!is_file_secure(password_file)) {
        std::cerr << "Password file permissions are too open. Only owner should have access." << std::endl;
        return 1;
    }

    // Read the password from the file
    std::ifstream secret_file(password_file);
    if (!secret_file.is_open()) {
        std::cerr << "Error opening the password file." << std::endl;
        return 1;
    }
    std::string password;
    if (!getline(secret_file, password) || password.empty()) {
        std::cerr << "Password file is empty or unreadable." << std::endl;
        return 1;
    }
    secret_file.close();

    // Email content (could also be parameterized)
    std::string email_data =
        "From: <" + mail_from + ">\r\n"
        "To: <" + mail_rcpt + ">\r\n"
        "Subject: Sending an email with CURL\r\n\r\n"
        "This is a pretty amazing email test, huh?\r\n";

    // Initialize libcurl
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl." << std::endl;
        secure_zero(&password[0], password.size());
        return 1;
    }

    // Set up libcurl options
    curl_easy_setopt(curl, CURLOPT_URL, "smtp://smtp.gmail.com:587");
    curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from.c_str());

    struct curl_slist *recipients = NULL;
    recipients = curl_slist_append(recipients, mail_rcpt.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    // Use STARTTLS
    curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    // Set up the email data as the read data for CURLOPT_READFUNCTION
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error." << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    // Clean up
    curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);

    // Securely erase password from memory
    secure_zero(&password[0], password.size());

    return 0;
}