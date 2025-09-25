#include <iostream>
#include <string>
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

int main() {
    // Initialize libcurl
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl." << std::endl;
        return 1;
    }

    // SMTP server details
    const std::string url = "smtps://smtp.gmail.com:465";
    const std::string username = "watchtower.test2023@gmail.com";
    std::string password; // To store the password

    // Read the password from the file "secret"
    std::ifstream secret_file("secret");
    if (!secret_file.is_open()) {
        std::cerr << "Error opening the 'secret' file." << std::endl;
        return 1;
    }

    getline(secret_file, password);
    secret_file.close();

    const std::string mail_from = "watchtower.test2023@gmail.com";
    const std::string mail_rcpt = "spam.mhered@gmail.com";

    // Email content
    const std::string email_data =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with CURL\r\n\r\n"
        "This is a pretty amazing email test, huh?\r\n";

    // Set up libcurl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from.c_str());

    struct curl_slist *recipients = NULL;
    recipients = curl_slist_append(recipients, mail_rcpt.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    // Set up the email data as the read data for CURLOPT_READFUNCTION
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    // Clean up
    curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);
    return 0;
}
