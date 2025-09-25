int main() {
    // ...

    // Calculate the size of the email data
    size_t email_size = email_data_str.size();

    // Set up the email data for the CURLOPT_READFUNCTION callback
    EmailData email_data = { email_data_str.c_str(), email_size };
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Set libcurl to print debug information to stderr
    curl_easy_setopt(curl, CURLOPT_DEBUGFUNCTION, curl_debug_callback);
    curl_easy_setopt(curl, CURLOPT_DEBUGDATA, nullptr);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

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
