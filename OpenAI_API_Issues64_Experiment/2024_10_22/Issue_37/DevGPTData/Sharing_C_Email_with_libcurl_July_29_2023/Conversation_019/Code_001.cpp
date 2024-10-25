#include <iostream>
#include <fstream>
#include <string>
#include <curl/curl.h>

// ... (rest of the code remains the same)

int main() {
    // Initialize libcurl
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl." << std::endl;
        return 1;
    }

    // Enable verbose output
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    // Rest of the code remains the same
    // ...

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
