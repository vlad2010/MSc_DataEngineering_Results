#include <iostream>
#include <fstream>
#include <string>
#include <curl/curl.h>

struct EmailData {
    std::string data;
    size_t offset;
};

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (email_data->offset >= email_data->data.size()) {
        return 0;
    }

    size_t len = std::min(size * nmemb, email_data->data.size() - email_data->offset);
    memcpy(ptr, email_data->data.data() + email_data->offset, len);
    email_data->offset += len;

    return len;
}

// ... (rest of the code remains the same)

int main() {
    // Initialize CURL library
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL." << std::endl;
        return 1;
    }

    // Email content setup
    std::string email_data_str =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with LIBCURL\r\n\r\n"
        "Ain't this a cool email test?\r\n";

    EmailData email_data = { email_data_str, 0 };

    // Configure CURL options
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // ... (other CURL setup)

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    // Clean up CURL resources
    curl_easy_cleanup(curl);
    return 0;
}