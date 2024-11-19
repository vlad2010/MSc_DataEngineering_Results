size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (email_data->size == 0) {
        return 0;
    }

    size_t len = std::min(size * nmemb, email_data->size);
    memcpy(ptr, email_data->data, len);
    email_data->data += len;
    email_data->size -= len;

    // Print the email body to console
    std::string body_chunk(static_cast<char*>(ptr), len);
    std::cout << "[Sent Body Chunk]: " << body_chunk;

    return len;
}
