#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <curl/curl.h>

struct EmailData {
    const char* data;
    size_t size;
    size_t offset; // Track progress instead of modifying pointer
};

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (email_data->offset >= email_data->size) {
        return 0;
    }

    size_t max_copy = size * nmemb;
    size_t remaining = email_data->size - email_data->offset;
    size_t len = std::min(max_copy, remaining);

    memcpy(ptr, email_data->data + email_data->offset, len);
    email_data->offset += len;

    // Debug output removed to avoid information disclosure (CWE-532)
    // If needed, enable only in secure, non-production builds
    // #ifdef DEBUG
    // std::cout << "[Sent Body]: " << std::string(static_cast<char*>(ptr), len) << std::endl;
    // #endif

    return len;
}

int main() {
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL." << std::endl;
        return 1;
    }

    // Email content
    const std::string email_data_str =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with LIBCURL\r\n\r\n"
        "Ain't this a cool email test?\r\n";

    EmailData email_data = { email_data_str.c_str(), email_data_str.size(), 0 };

    // Set up the email data for the CURLOPT_READFUNCTION callback
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // ... (other necessary curl options, e.g., URL, authentication, etc.)

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    curl_easy_cleanup(curl);
    return 0;
}