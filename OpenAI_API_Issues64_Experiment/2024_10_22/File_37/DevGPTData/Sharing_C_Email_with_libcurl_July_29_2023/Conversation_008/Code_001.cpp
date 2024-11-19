#include <iostream>
#include <string>
#include <cstring>  // Include this header for memcpy()
#include <curl/curl.h>

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
    if (data->empty()) {
        return 0;
    }

    size_t len = std::min(size * nmemb, data->size());
    memcpy(ptr, data->data(), len);
    data->erase(0, len);
    return len;
}

// ... rest of the code ...
