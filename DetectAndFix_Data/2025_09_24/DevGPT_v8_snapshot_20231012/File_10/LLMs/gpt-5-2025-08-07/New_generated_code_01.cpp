#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>
#include <curl/curl.h>

struct EmailData {
    const char* data;   // points into a stable std::string buffer
    size_t size;        // total size
    size_t sent;        // bytes already sent
};

// Safe read callback with overflow guard
static size_t read_data_callback(char* ptr, size_t size, size_t nmemb, void* user_data) {
    if (!ptr || !user_data) return 0;
    if (size == 0 || nmemb == 0) return 0;

    // Guard against size * nmemb overflow
    size_t max_chunk;
    if (nmemb > SIZE_MAX / size) {
        max_chunk = SIZE_MAX;
    } else {
        max_chunk = size * nmemb;
    }

    EmailData* ed = static_cast<EmailData*>(user_data);
    if (ed->sent >= ed->size) return 0;

    size_t remaining = ed->size - ed->sent;
    size_t to_copy = std::min(remaining, max_chunk);

    // memcpy into libcurl's buffer
    std::memcpy(ptr, ed->data + ed->sent, to_copy);
    ed->sent += to_copy;
    return to_copy;
}

struct ResponseBuffer {
    std::string data;
    size_t limit; // cap to avoid unbounded memory growth
};

// Write callback that caps memory use and does not write to stdout
static size_t write_data_callback(char* ptr, size_t size, size_t nmemb, void* user_data) {
    if (!ptr || !user_data) return 0;
    if (size == 0 || nmemb == 0) return 0;

    // Guard overflow in multiplication
    size_t total;
    if (nmemb > SIZE_MAX / size) {
        total = SIZE_MAX;
    } else {
        total = size * nmemb;
    }

    ResponseBuffer* rb = static_cast<ResponseBuffer*>(user_data);
    // Append up to the limit; discard the rest but report consumed bytes to libcurl.
    size_t can_append = (rb->data.size() >= rb->limit) ? 0 : (rb->limit - rb->data.size());
    size_t will_append = std::min(can_append, total);
    if (will_append > 0) {
        rb->data.append(ptr, will_append);
    }
    // Returning "total" indicates we've consumed the data even if we discarded some.
    return total;
}

int main() {
    // Initialize libcurl globally
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
        std::cerr << "Error initializing libcurl global state." << std::endl;
        return 1;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl easy handle." << std::endl;
        curl_global_cleanup();
        return 1;
    }

    char errbuf[CURL_ERROR_SIZE] = {0};
    CURLcode rc;

    // Email (request body) content
    const std::string email_data_str = "This is the custom message sent in the request body.";
    EmailData email_data{ email_data_str.data(), email_data_str.size(), 0 };

    // Prepare response buffer with a sane cap (e.g., 1 MB)
    ResponseBuffer response{ std::string(), 1024 * 1024 };

    // Optional: add a Content-Type header
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: text/plain; charset=utf-8");
    // Optional: avoid "Expect: 100-continue" delays for small payloads
    headers = curl_slist_append(headers, "Expect:");

    // Set common options with error checks
    rc = curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
    if (rc != CURLE_OK) { std::cerr << "setopt ERRORBUFFER failed\n"; }

    // Use HTTPS and restrict protocols to HTTPS
    rc = curl_easy_setopt(curl, CURLOPT_URL, "https://httpbin.org/post");
    if (rc != CURLE_OK) { std::cerr << "Failed to set URL: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTPS);
    if (rc != CURLE_OK) { std::cerr << "Failed to restrict protocols: " << curl_easy_strerror(rc) << "\n"; }

    // Enforce TLS certificate and hostname validation (CWE-295)
    rc = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    if (rc != CURLE_OK) { std::cerr << "Failed to set SSL_VERIFYPEER: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    if (rc != CURLE_OK) { std::cerr << "Failed to set SSL_VERIFYHOST: " << curl_easy_strerror(rc) << "\n"; }
    // Note: Ensure system has a proper CA bundle. If needed:
    // curl_easy_setopt(curl, CURLOPT_CAINFO, "/path/to/cacert.pem");

    // POST using a read callback and explicit Content-Length
    rc = curl_easy_setopt(curl, CURLOPT_POST, 1L);
    if (rc != CURLE_OK) { std::cerr << "Failed to set POST: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    if (rc != CURLE_OK) { std::cerr << "Failed to set READFUNCTION: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
    if (rc != CURLE_OK) { std::cerr << "Failed to set READDATA: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(email_data.size));
    if (rc != CURLE_OK) { std::cerr << "Failed to set POSTFIELDSIZE_LARGE: " << curl_easy_strerror(rc) << "\n"; }

    // Avoid writing response to stdout; capture it safely (CWE-532)
    rc = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data_callback);
    if (rc != CURLE_OK) { std::cerr << "Failed to set WRITEFUNCTION: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    if (rc != CURLE_OK) { std::cerr << "Failed to set WRITEDATA: " << curl_easy_strerror(rc) << "\n"; }

    // Attach headers
    rc = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (rc != CURLE_OK) { std::cerr << "Failed to set HTTPHEADER: " << curl_easy_strerror(rc) << "\n"; }

    // Timeouts and robustness (CWE-400)
    rc = curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    if (rc != CURLE_OK) { std::cerr << "Failed to set CONNECTTIMEOUT: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    if (rc != CURLE_OK) { std::cerr << "Failed to set TIMEOUT: " << curl_easy_strerror(rc) << "\n"; }
    rc = curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L); // safer in multi-threaded apps
    if (rc != CURLE_OK) { std::cerr << "Failed to set NOSIGNAL: " << curl_easy_strerror(rc) << "\n"; }

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the request. Error: "
                  << (errbuf[0] ? errbuf : curl_easy_strerror(res)) << std::endl;
    } else {
        std::cout << "Request sent successfully." << std::endl;
        // Use response.data as needed (avoid logging sensitive data)
        // std::cout << "Response (truncated if large): " << response.data << std::endl;
    }

    // Clean up
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return (res == CURLE_OK) ? 0 : 1;
}