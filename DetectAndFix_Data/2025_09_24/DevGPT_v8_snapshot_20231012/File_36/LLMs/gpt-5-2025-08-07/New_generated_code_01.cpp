#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <curl/curl.h>

struct UploadCtx {
    const char* data;
    size_t len;
    size_t pos;
};

static size_t read_data_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    UploadCtx* ctx = static_cast<UploadCtx*>(userdata);
    if (!ctx || !ctx->data) return 0;

    size_t max_bytes = size * nmemb;
    if (ctx->pos >= ctx->len) return 0; // EOF

    size_t remaining = ctx->len - ctx->pos;
    size_t to_copy = remaining < max_bytes ? remaining : max_bytes;

    memcpy(ptr, ctx->data + ctx->pos, to_copy);
    ctx->pos += to_copy;
    return to_copy;
}

int main() {
    CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (rc != CURLE_OK) {
        std::cerr << "curl_global_init failed: " << curl_easy_strerror(rc) << std::endl;
        return 1;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error initializing libcurl." << std::endl;
        curl_global_cleanup();
        return 1;
    }

    // Configuration: prefer environment variables over hard-coding
    // For Gmail, prefer XOAUTH2 (GMAIL_OAUTH2_TOKEN). If unavailable, allow app password via env var.
    // Do NOT read plaintext password from files.
    const char* username = std::getenv("GMAIL_USERNAME");
    const char* xoauth2_token = std::getenv("GMAIL_OAUTH2_TOKEN");     // Preferred
    const char* app_password  = std::getenv("GMAIL_APP_PASSWORD");      // Fallback (less secure)

    if (!username || (!xoauth2_token && !app_password)) {
        std::cerr << "Missing credentials. Set GMAIL_USERNAME and either GMAIL_OAUTH2_TOKEN (preferred) or GMAIL_APP_PASSWORD." << std::endl;
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        return 1;
    }

    // Server and mail parameters
    // Use implicit TLS (smtps://) to avoid STARTTLS downgrade risks.
    const std::string url       = "smtps://smtp.gmail.com:465";
    const std::string mail_from = username; // sender
    const std::string mail_rcpt = "recipient@example.com"; // change to your recipient

    // Email content (keep constants or sanitize if any field comes from untrusted input)
    const std::string email_data =
        "From: " + std::string(username) + "\r\n"
        "To: recipient@example.com\r\n"
        "Subject: Sending an email with cURL (secure)\r\n"
        "\r\n"
        "This is a secure email test.\r\n";

    UploadCtx upload_ctx{ email_data.c_str(), email_data.size(), 0 };

    // Helper macro to fail fast on setopt errors (CWE-252)
    #define SETOPT(opt, val) do { \
        rc = curl_easy_setopt(curl, opt, val); \
        if (rc != CURLE_OK) { \
            std::cerr << "curl_easy_setopt failed for " #opt ": " << curl_easy_strerror(rc) << std::endl; \
            goto cleanup; \
        } \
    } while (0)

    // Basic URL and authentication
    SETOPT(CURLOPT_URL, url.c_str());
    SETOPT(CURLOPT_USERNAME, username);

    if (xoauth2_token) {
        // Preferred: OAuth 2.0 (XOAUTH2). No password sent.
        SETOPT(CURLOPT_XOAUTH2_BEARER, xoauth2_token);
    } else {
        // Fallback: App password from env var (not ideal, but avoids plaintext files)
        SETOPT(CURLOPT_PASSWORD, app_password);
    }

    // SMTP specifics
    SETOPT(CURLOPT_MAIL_FROM, mail_from.c_str());
    struct curl_slist* recipients = nullptr;
    recipients = curl_slist_append(recipients, mail_rcpt.c_str());
    if (!recipients) {
        std::cerr << "Failed to set recipient." << std::endl;
        goto cleanup;
    }
    SETOPT(CURLOPT_MAIL_RCPT, recipients);

    // Upload the message body
    SETOPT(CURLOPT_READFUNCTION, read_data_callback);
    SETOPT(CURLOPT_READDATA, &upload_ctx);
    SETOPT(CURLOPT_UPLOAD, 1L);

    // TLS hardening
    SETOPT(CURLOPT_SSL_VERIFYPEER, 1L);
    SETOPT(CURLOPT_SSL_VERIFYHOST, 2L);
    SETOPT(CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2); // Or TLSv1_3 if supported
    // With smtps:// we don't need CURLOPT_USE_SSL; do not set it to TRY (avoid downgrade).

    // Timeouts and robustness
    SETOPT(CURLOPT_CONNECTTIMEOUT, 10L);
    SETOPT(CURLOPT_TIMEOUT, 60L);
    SETOPT(CURLOPT_NOSIGNAL, 1L);
    SETOPT(CURLOPT_TCP_KEEPALIVE, 1L);
    SETOPT(CURLOPT_PROTOCOLS, CURLPROTO_SMTPS); // Restrict protocol

    // Send the email
    rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        std::cerr << "Failed to send email: " << curl_easy_strerror(rc) << std::endl;
        goto cleanup;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

cleanup:
    if (recipients) curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    return (rc == CURLE_OK) ? 0 : 1;

    #undef SETOPT
}