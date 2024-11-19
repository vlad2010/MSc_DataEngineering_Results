// Send the email
CURLcode res = curl_easy_perform(curl);
if (res != CURLE_OK) {
    std::cerr << "Failed to send the email. Error: " << curl_easy_strerror(res) << std::endl;
    if (res == CURLE_SSL_CACERT || res == CURLE_SSL_PEER_CERTIFICATE) {
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // Enable verbose output
        curl_easy_perform(curl);
    }
} else {
    std::cout << "Email sent successfully." << std::endl;
}
