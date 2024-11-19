#include <iostream>
#include <string>
#include <curl/curl.h>
#include <libxml/HTMLparser.h>
#include <libxml/xpath.h>

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
            // Parse the HTML response
            htmlDocPtr doc = htmlReadDoc(reinterpret_cast<const xmlChar*>(response.c_str()), nullptr, nullptr, HTML_PARSE_NOERROR | HTML_PARSE_NOWARNING);
            if (doc) {
                // Create an XPath context
                xmlXPathContextPtr xpathCtx = xmlXPathNewContext(doc);
                if (xpathCtx) {
                    // Register the namespaces used in the XPath expressions
                    xmlXPathRegisterNs(xpathCtx, BAD_CAST "html", BAD_CAST "http://www.w3.org/1999/xhtml");

                    // Evaluate the XPath expression to select the table rows
                    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(BAD_CAST "//html:table[@id='public_holidays']/html:tbody/html:tr", xpathCtx);
                    if (xpathObj) {
                        // Iterate over the selected table rows
                        xmlNodeSetPtr nodes = xpathObj->nodesetval;
                        for (int i = 0; i < nodes->nodeNr; ++i) {
                            // Extract the date and holiday name from each row
                            xmlNodePtr node = nodes->nodeTab[i];
                            xmlNodePtr dateNode = xmlFirstElementChild(node);
                            xmlNodePtr holidayNode = xmlNextElementSibling(dateNode);

                            // Print the date and holiday name in Markdown table format
                            std::cout << "| " << xmlNodeGetContent(dateNode) << " | " << xmlNodeGetContent(holidayNode) << " |" << std::endl;
                        }

                        // Cleanup XPath object
                        xmlXPathFreeObject(xpathObj);
                    }

                    // Cleanup XPath context
                    xmlXPathFreeContext(xpathCtx);
                }

                // Cleanup parsed document
                xmlFreeDoc(doc);
            }
        }

        // Clean up curl
        curl_easy_cleanup(curl);
    }

    return 0;
}
