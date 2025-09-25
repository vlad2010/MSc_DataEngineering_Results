#include <iostream>
#include <string>
#include <algorithm>
#include <cstring>
#include <limits>

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
    // Check for null pointers
    if (!ptr || !data) {
        return 0;
    }

    if (size == 0 || nmemb == 0 || data->empty()) {
        return 0;
    }

    // Prevent overflow in size * nmemb
    if (size > 0 && nmemb > std::numeric_limits<size_t>::max() / size) {
        // size * nmemb would overflow
        return 0;
    }
    size_t buffer_size = size * nmemb;

    // Only copy as much as both the buffer and the data can handle
    size_t len = std::min(buffer_size, data->size());
    if (len > 0) {
        memcpy(ptr, data->data(), len);
        data->erase(0, len);
    }

    // [SECURITY] Remove or restrict debug output to avoid leaking sensitive data
    // std::cout << "[Sent Body]: " << std::string(static_cast<char*>(ptr), len) << std::endl;
    // If absolutely necessary, log only in a secure, redacted, or controlled environment

    return len;
}