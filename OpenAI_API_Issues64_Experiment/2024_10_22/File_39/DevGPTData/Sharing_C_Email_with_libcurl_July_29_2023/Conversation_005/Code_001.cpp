#include <iostream>
#include <fstream>
#include <httplib.h>

int main() {
    // SMTP server details
    const std::string smtp_host = "smtp.gmail.com";
    const int smtp_port = 465;
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

    // Email content
    const std::string email_data =
        "From: Watchtower Tester <watchtower.test2023@gmail.com>\r\n"
        "To: SPAM <spam.mhered@gmail.com>\r\n"
        "Subject: Sending an email with cpp-httplib\r\n\r\n"
        "This is a pretty amazing email test, huh?\r\n";

    // Create an HTTPS client and set up SSL options
    httplib::SSLClient smtp_client(smtp_host, smtp_port);
    smtp_client.enable_server_certificate_verification(true);

    // Authentication using base64 encoding
    std::string auth_str = username + ":" + password;
    smtp_client.set_basic_auth(username.c_str(), password.c_str());

    // Send the email
    const auto res = smtp_client.Post("/smtps://smtp.gmail.com:465", email_data, "message/rfc822");
    if (!res || res->status != 200) {
        std::cerr << "Failed to send the email." << std::endl;
    } else {
        std::cout << "Email sent successfully." << std::endl;
    }

    return 0;
}
