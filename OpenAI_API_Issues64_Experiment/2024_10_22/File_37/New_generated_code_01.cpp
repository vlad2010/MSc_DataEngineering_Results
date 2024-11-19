#include <iostream>
#include <string>
#include <algorithm>
#include <cstring>

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
    // Ensure that data is not null
    if (data == nullptr || data->empty()) {
        return 0;
    }

    // Calculate buffer size
    size_t bufferSize = size * nmemb;
    
    // Check that the buffer can accommodate the data to be copied
    if (ptr == nullptr || bufferSize == 0) {
        return 0;
    }
    
    // Determine the number of bytes to write
    size_t len = std::min(bufferSize, data->size());

    // Safely copy data to the provided buffer
    memcpy(ptr, data->data(), len);
    data->erase(0, len);

    // Comment out or remove sensitive information logging
    // std::cout << "[Sent Body]: " << std::string(static_cast<char*>(ptr), len) << std::endl;
    
    return len;
}