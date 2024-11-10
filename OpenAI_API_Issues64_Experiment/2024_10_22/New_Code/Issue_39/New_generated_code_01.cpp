#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // for getenv
#include <vector>
#include <curl/curl.h>

struct EmailData {
    const char* data;
    size_t size;
    size_t bytes_read;
};

static size_t read_data_callback(void *ptr, size_t size, size_t nmemb, void *userp) {
    EmailData *email_data = static_cast<EmailData *>(userp);
    size_t room = size * nmemb;
    size_t remain = email_data->size - email_data->bytes_read;
    size_t write_size = std::min(room, remain);
    
    if(remain == 0) {
        return 0;
    }

    memcpy(ptr, email_data->data + email_data->bytes_read, write_size);
    email_data->bytes_read += write_size;
    
    return write_size;
}

bool get_password(std::string &password) {
    const char* password_env = getenv("SMTP_PASSWORD");
    if (password_env) {
        password = password_env;
        return true;
    }
    return false;
}

class EmailSender {
public:
    EmailSender() : curl(curl_easy_init()) {
        if (!curl) {
            std::cerr << "Error initializing libcurl." << std::endl;
            return;
        }
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        const std::string url = "smtps://smtp.gmail.com:465";
        const std::string username = "watchtower.test2023@gmail.com";

        if (!get_password(password)) {
            std::cerr << "Error obtaining the password from environment variables." << std::endl;
            return;
        }

        const std::string mail_from = "watchtower.test2023@gmail.com";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
        curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, mail_from.c_str());

        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    }

    ~EmailSender() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

    bool sendEmail(const std::vector<std::string>& recipients, const std::string& subject, const std::string& body) {
        std::string email_content = "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n";
        email_content += "To: ";
        for (size_t i = 0; i < recipients.size(); ++i) {
            email_content += recipients[i];
            if (i < recipients.size() - 1) {
                email_content += ",";
            }
        }
        email_content += "\r\n";
        email_content += "Subject: " + subject + "\r\n";
        email_content += "\r\n" + body;

        EmailData email_data = { email_content.c_str(), email_content.size(), 0 };
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data_callback);
        curl_easy_setopt(curl, CURLOPT_READDATA, &email_data);
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

        struct curl_slist *recipients_list = NULL;
        for (const std::string& recipient : recipients) {
            recipients_list = curl_slist_append(recipients_list, recipient.c_str());
        }
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients_list);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
            curl_slist_free_all(recipients_list);
            return false;
        }

        curl_slist_free_all(recipients_list);
        return true;
    }

private:
    CURL *curl;
    std::string password;
};

int main() {
    EmailSender emailSender;

    std::vector<std::string> recipients = {"recipient1@example.com", "recipient2@example.com"};
    std::string subject = "Test Subject";
    std::string body = "This is the body of the email.";

    if (emailSender.sendEmail(recipients, subject, body)) {
        std::cout << "Email sent successfully." << std::endl;
    } else {
        std::cerr << "Failed to send the email." << std::endl;
    }

    return 0;
}