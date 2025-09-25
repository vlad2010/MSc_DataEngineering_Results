int curl_debug_callback(CURL *curl, curl_infotype type, char *data, size_t size, void *userptr) {
    (void)userptr; // Avoid unused parameter warning

    if (type == CURLINFO_TEXT) {
        std::cout << "[CURL INFO]: ";
    } else if (type == CURLINFO_HEADER_IN || type == CURLINFO_HEADER_OUT) {
        std::cout << "[CURL HEADER]: ";
    } else if (type == CURLINFO_DATA_IN || type == CURLINFO_DATA_OUT) {
        std::cout << "[CURL DATA]: ";
    } else if (type == CURLINFO_SSL_DATA_IN || type == CURLINFO_SSL_DATA_OUT) {
        std::cout << "[CURL SSL DATA]: ";
    } else if (type == CURLINFO_END) {
        std::cout << "[CURL END]: ";
    } else {
        // Unknown type, print as text
        std::cout << "[CURL UNKNOWN]: ";
    }

    std::cout.write(data, size);
    std::cout << std::endl;
    return 0;
}
