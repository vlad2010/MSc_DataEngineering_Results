#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <curl/curl.h>

struct EmailData {
    const char* data;
    size_t size;
};

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (email_data->size == 0) {
        return 0;
    }

    size_t len = std::min(size * nmemb, email_data->size);
    memcpy(ptr, email_data->data, len);
    email_data->data += len;
    email_data->size -= len;

    // Add custom debug information to print the body being sent
    std::cout << "[Sent Body]: " << std::string(static_cast<char*>(ptr), len) << std::endl;

    return len;
}

// ... (rest of the code remains the same)

int main() {
    // ...

    // Email content
    const std::string email_data_str =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with LIBCURL\r\n\r\n"
        "Ain't this a cool email test?\r\n";

    EmailData email_data = { email_data_str.c_str(), email_data_str.size() };

    // Set up the email data for the CURLOPT_READFUNCTION callback
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // ...

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    // ...
    return 0;
}
