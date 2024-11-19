#include <iostream>
#include <string>
#include <curl/curl.h>

// Callback function to write fetched data into a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

int main() {
    // Variables for storing user input
    std::string country_code;
    int year;

    // Get user input
    std::cout << "Enter the country code (e.g., us, uk): ";
    std::cin >> country_code;
    std::cout << "Enter the year: ";
    std::cin >> year;

    // Create the URL string
    std::string url = "https://www.officeholidays.com/countries/" + country_code + "/" + std::to_string(year);

    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (curl) {
        // Set the URL to fetch
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Create a string to store the fetched data
        std::string response;

        // Set the callback function to write the data into the string
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Perform the request
        CURLcode res = curl_easy_perform(curl);

        // Check for errors
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            // Print the fetched data
            std::cout << response << std::endl;
        }

        // Clean up curl
        curl_easy_cleanup(curl);
    }

    return 0;
}
