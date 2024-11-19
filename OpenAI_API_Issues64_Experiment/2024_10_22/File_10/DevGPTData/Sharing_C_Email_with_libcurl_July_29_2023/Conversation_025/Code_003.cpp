int curl_debug_callback(CURL *curl, curl_infotype type, char *data, size_t size, void *userptr) {
    std::cout << "[CURL DEBUG]: ";
    std::cout.write(data, size);
    std::cout << std::endl;
    return 0;
}
