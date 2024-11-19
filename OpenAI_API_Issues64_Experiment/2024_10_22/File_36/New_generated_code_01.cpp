#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
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

std::string get_password_from_env() {
    const char* pwd = std::getenv("SMTP_PASSWORD");
    if (!pwd) {
        throw std::runtime_error("SMTP_PASSWORD environment variable not set.");
    }
    return std::string(pwd);
}

int main() {
    try {
        // Initialize libcurl
        CURL *curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Error initializing libcurl." << std::endl;
            return 1;
        }

        // SMTP server details
        const std::string url = "smtp://smtp.gmail.com:587";
        const std::string username = "watchtower.test2023@gmail.com";
        std::string password = get_password_from_env();  // Use environment variable to store password

        const std::string mail_from = "watchtower.test2023@gmail.com";
        const std::string mail_rcpt = "spam.mhered@gmail.com";

        // Email content
        const std::string email_data =
            "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
            "To: SPAM <spam.mhered@gmail.com>\r\n"
            "Subject: Sending an email with CURL\r\n\r\n"
            "This is a pretty amazing email test, huh?\r\n";

        // Set up libcurl options
        CURLcode res;
        res = curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from.c_str());
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }

        struct curl_slist *recipients = NULL;
        recipients = curl_slist_append(recipients, mail_rcpt.c_str());
        res = curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }

        // Set up STARTTLS
        res = curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }

        // Set up the email data as the read data for CURLOPT_READFUNCTION
        res = curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }
        res = curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
        if (res != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(res)); }

        // Send the email
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Email sent successfully." << std::endl;
        }

        // Clean up
        curl_slist_free_all(recipients);
        curl_easy_cleanup(curl);
    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}