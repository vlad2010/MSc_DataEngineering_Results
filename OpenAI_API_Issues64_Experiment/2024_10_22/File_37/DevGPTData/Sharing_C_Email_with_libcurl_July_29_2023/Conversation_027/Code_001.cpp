#include <iostream>
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

    return len;
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
    EmailData email_data = { email_data_str.c_str(), email_data_str.size() };
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Set URL to send the request
    curl_easy_setopt(curl, CURLOPT_URL, "http://httpbin.org/post"); // Replace with your desired URL

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
