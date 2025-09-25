#include <iostream>
#include <cstring>
#include <string>
#include <curl/curl.h>

struct EmailData {
    const char* data;
    size_t size;
    size_t total_size; // Track original size for safety
};

// Callback function to read data from the email body
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    // Validate input parameters
    if (!ptr || !user_data) {
        return 0;
    }
    
    // Check for integer overflow
    size_t buffer_size = size * nmemb;
    if (size != 0 && buffer_size / size != nmemb) {
        return 0; // Integer overflow detected
    }
    
    EmailData *email_data = static_cast<EmailData *>(user_data);
    
    // Validate email_data state
    if (!email_data->data || email_data->size == 0) {
        return 0;
    }
    
    size_t len = std::min(buffer_size, email_data->size);
    memcpy(ptr, email_data->data, len);
    email_data->data += len;
    email_data->size -= len;

    return len;
}

int main() {
    // Initialize libcurl globally (thread-safe initialization)
    CURLcode global_init_res = curl_global_init(CURL_GLOBAL_ALL);
    if (global_init_res != CURLE_OK) {
        std::cerr << "Error initializing libcurl globally: " 
                  << curl_easy_strerror(global_init_res) << std::endl;
        return 1;
    }

    // Initialize libcurl easy handle
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl easy handle." << std::endl;
        curl_global_cleanup();
        return 1;
    }

    int return_code = 0;
    
    try {
        // Email content
        const std::string email_data_str = "This is the custom message sent in the request body.";

        // Set up the email data for the CURLOPT_READFUNCTION callback
        EmailData email_data = { 
            email_data_str.c_str(), 
            email_data_str.size(),
            email_data_str.size()  // Store original size
        };
        
        // Use HTTPS instead of HTTP for secure transmission
        CURLcode setopt_res = curl_easy_setopt(curl, CURLOPT_URL, "https://httpbin.org/post");
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set URL: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Explicitly set this as a POST request
        setopt_res = curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set POST: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Set the size of the POST data
        setopt_res = curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(email_data_str.size()));
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set POST field size: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Set read callback and data
        setopt_res = curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set read function: " + std::string(curl_easy_strerror(setopt_res)));
        }

        setopt_res = curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set read data: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Enable SSL/TLS certificate verification (security best practice)
        setopt_res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set SSL verify peer: " + std::string(curl_easy_strerror(setopt_res)));
        }

        setopt_res = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set SSL verify host: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Set timeout to prevent hanging indefinitely
        setopt_res = curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        if (setopt_res != CURLE_OK) {
            throw std::runtime_error("Failed to set timeout: " + std::string(curl_easy_strerror(setopt_res)));
        }

        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to send the request. Error: " << curl_easy_strerror(res) << std::endl;
            return_code = 1;
        } else {
            std::cout << "Request sent successfully." << std::endl;
            
            // Get response code for additional validation
            long response_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cout << "HTTP Response Code: " << response_code << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return_code = 1;
    }

    // Clean up
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    
    return return_code;
}