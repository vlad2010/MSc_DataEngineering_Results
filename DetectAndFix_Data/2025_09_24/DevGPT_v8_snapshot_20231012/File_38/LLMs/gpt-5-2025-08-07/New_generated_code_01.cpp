#include <curl/curl.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>

struct EmailData {
    const char* buf;     // Pointer to immutable email data buffer
    size_t total;        // Total size of the buffer
    size_t sent;         // Bytes already sent
    bool debug;          // Whether to log sanitized chunks
};

// Sanitized (and truncated) logging of data to avoid CWE-117 and CWE-532.
// Not enabled by default; toggled by DEBUG_EMAIL_BODY environment variable.
static void debug_log_chunk(const char* data, size_t len) {
    const size_t kMaxLog = 256; // truncate to avoid large logs
    size_t to_log = len < kMaxLog ? len : kMaxLog;
    std::string sanitized;
    sanitized.reserve(to_log * 4);
    for (size_t i = 0; i < to_log; ++i) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c >= 0x20 && c <= 0x7E) {
            sanitized.push_back(static_cast<char>(c));
        } else {
            char esc[5];
            std::snprintf(esc, sizeof(esc), "\\x%02X", c);
            sanitized += esc;
        }
    }
    std::cerr << "[DEBUG] Sent body chunk (" << to_log << " bytes, sanitized): " << sanitized << std::endl;
}

// Callback function to read data from the email body securely
size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *user_data) {
    EmailData *email_data = static_cast<EmailData *>(user_data);
    if (!email_data || !ptr) {
        return CURL_READFUNC_ABORT;
    }

    // Nothing left to send
    if (email_data->sent >= email_data->total) {
        return 0;
    }

    // Prevent size*nmemb overflow (CWE-190)
    if (nmemb != 0 && size > std::numeric_limits<size_t>::max() / nmemb) {
        return CURL_READFUNC_ABORT;
    }

    size_t max_chunk = size * nmemb;
    size_t remaining = email_data->total - email_data->sent;
    size_t len = remaining < max_chunk ? remaining : max_chunk;

    // Copy the next chunk
    std::memcpy(ptr, email_data->buf + email_data->sent, len);
    email_data->sent += len;

    // Optional, sanitized, truncated debug logging (disabled by default)
    if (email_data->debug) {
        debug_log_chunk(static_cast<const char*>(ptr), len);
    }

    return len;
}

int main() {
    CURLcode global_init = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (global_init != CURLE_OK) {
        std::cerr << "curl_global_init failed: " << curl_easy_strerror(global_init) << std::endl;
        return 1;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl_easy_init failed" << std::endl;
        curl_global_cleanup();
        return 1;
    }

    // Environment configuration to avoid hard-coded secrets (CWE-798)
    const char* smtp_url = std::getenv("SMTP_URL");       // e.g., "smtps://smtp.gmail.com:465" or "smtp://smtp.gmail.com:587"
    const char* smtp_user = std::getenv("SMTP_USER");     // SMTP username
    const char* smtp_pass = std::getenv("SMTP_PASS");     // SMTP password or app password
    const char* debug_env = std::getenv("DEBUG_EMAIL_BODY");

    // Reasonable defaults if SMTP_URL not set: prefer SMTPS (implicit TLS)
    std::string url = smtp_url ? smtp_url : "smtps://smtp.gmail.com:465";

    // Basic email addressing (do not hard-code in production)
    const char* mail_from = "watchtower.test2023@gmail.com";
    const char* rcpt_to   = "spam.mhered@gmail.com";

    // Email content (ensure CRLF per SMTP)
    const std::string email_data_str =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with LIBCURL\r\n"
        "\r\n"
        "Ain't this a cool email test?\r\n";

    // Build email data struct with offset tracking
    EmailData email_data{
        email_data_str.c_str(),
        email_data_str.size(),
        0,
        debug_env && std::string(debug_env) == "1" // only log when explicitly enabled
    };

    // Configure libcurl
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L); // required for SMTP payloads

    // Enforce TLS and verify certificates to avoid CWE-319 and CWE-295
    // If URL is smtp:// (port 587), STARTTLS will be used; for smtps:// (port 465), implicit TLS is used.
    curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    // Optionally enforce a minimum TLS version (requires a libcurl with SSL backend supporting it)
    curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);

    // Timeouts and safety
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    // Credentials (avoid hard-coding; rely on env variables)
    if (smtp_user && *smtp_user) {
        curl_easy_setopt(curl, CURLOPT_USERNAME, smtp_user);
    }
    if (smtp_pass && *smtp_pass) {
        curl_easy_setopt(curl, CURLOPT_PASSWORD, smtp_pass);
    }

    // Envelope addresses
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from);
    struct curl_slist* recipients = nullptr;
    recipients = curl_slist_append(recipients, rcpt_to);
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    // Provide the read callback and context
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);

    // Avoid verbose mode in production (it can leak sensitive data too)
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

    // Send the email
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    // Cleanup
    if (recipients) {
        curl_slist_free_all(recipients);
    }
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return 0;
}