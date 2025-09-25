#include <iostream>
#include <string>
#include <algorithm>
#include <curl/curl.h>

struct EmailData {
    const std::string* data; // Pointer to the string, not its buffer
    size_t offset;           // Current offset in the string
};

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    size_t buffer_size = size * nmemb;
    size_t remaining = email_data->data->size() - email_data->offset;

    if (remaining == 0) {
        return 0;
    }

    size_t to_copy = std::min(buffer_size, remaining);
    memcpy(ptr, email_data->data->data() + email_data->offset, to_copy);
    email_data->offset += to_copy;

    return to_copy;
}

int main() {
    // Initialize libcurl
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl." << std::endl;
        return 1;
    }

    // Email content
    const std::string email_data_str = "This is the custom message sent in the request body.";

    // Set up the email data for the CURLOPT_READFUNCTION callback
    EmailData email_data = { &email_data_str, 0 };
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Set URL to send the request
    // Use HTTPS for security (CWE-327)
    curl_easy_setopt(curl, CURLOPT_URL, "https://httpbin.org/post"); // Replace with your desired URL

    // Set other options as needed (e.g., headers, authentication, etc.)

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the request. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Request sent successfully." << std::endl;
    }

    // Clean up
    curl_easy_cleanup(curl);
    return 0;
}