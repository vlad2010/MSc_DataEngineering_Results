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
