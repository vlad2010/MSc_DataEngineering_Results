#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <curl/curl.h>
#include <libxml/HTMLparser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

// Callback function to write fetched data into a string
size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *response)
{
    size_t total_size = size * nmemb;
    // Prevent integer overflow
    if (size != 0 && total_size / size != nmemb) {
        return 0;
    }
    
    // Limit response size to prevent memory exhaustion (10MB limit)
    const size_t MAX_RESPONSE_SIZE = 10 * 1024 * 1024;
    if (response->size() + total_size > MAX_RESPONSE_SIZE) {
        return 0;
    }
    
    response->append(static_cast<char *>(contents), total_size);
    return total_size;
}

// Input validation function for country code
bool isValidCountryCode(const std::string& country) {
    // Allow only alphabetic characters and limit length
    std::regex country_regex("^[a-zA-Z]{2,3}$");
    return std::regex_match(country, country_regex);
}

// Input validation function for year
bool isValidYear(int year) {
    // Reasonable year range
    return year >= 1900 && year <= 2100;
}

std::vector<std::pair<std::string, std::string>> fetchBankHolidays(const std::string& country, int year)
{
    std::vector<std::pair<std::string, std::string>> bankHolidays;

    // Validate inputs
    if (!isValidCountryCode(country) || !isValidYear(year)) {
        std::cerr << "Invalid input parameters" << std::endl;
        return bankHolidays;
    }

    // Create the URL string with validated inputs
    std::string url = "https://www.officeholidays.com/countries/" + country + "/" + std::to_string(year);

    // Initialize libcurl
    CURL *curl = curl_easy_init();
    if (curl)
    {
        // Set the URL to fetch
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Create a string to store the fetched data
        std::string response;

        // Set the callback function to write the data into the string
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Security: Enable SSL/TLS certificate verification
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

        // Set timeouts to prevent hanging
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

        // Follow redirects but limit them
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);

        // Restrict protocols to HTTPS only
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTPS);
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTPS);

        // Perform the request
        CURLcode res = curl_easy_perform(curl);

        // Check for errors
        if (res != CURLE_OK)
        {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        else
        {
            // Parse the HTML response
            htmlDocPtr doc = htmlReadDoc(reinterpret_cast<const xmlChar *>(response.c_str()), 
                                        nullptr, nullptr, 
                                        HTML_PARSE_NOERROR | HTML_PARSE_NOWARNING | HTML_PARSE_NOBLANKS);
            if (doc)
            {
                // Create an XPath context
                xmlXPathContextPtr xpathCtx = xmlXPathNewContext(doc);
                if (xpathCtx)
                {
                    // Register the namespaces used in the XPath expressions
                    xmlXPathRegisterNs(xpathCtx, BAD_CAST "html", BAD_CAST "http://www.w3.org/1999/xhtml");

                    // Evaluate the XPath expression to select the table rows
                    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(
                        BAD_CAST "//table[@class='country-table']//tr[position()>1]", xpathCtx);
                    
                    if (xpathObj && xpathObj->nodesetval)
                    {
                        // Iterate over the selected table rows
                        xmlNodeSetPtr nodes = xpathObj->nodesetval;
                        for (int i = 0; i < nodes->nodeNr && i < 100; ++i) // Limit iterations
                        {
                            xmlNodePtr node = nodes->nodeTab[i];
                            if (!node) continue;

                            xmlNodePtr dateNode = xmlFirstElementChild(node);
                            if (!dateNode) continue;

                            xmlNodePtr holidayNode = xmlNextElementSibling(dateNode);
                            if (!holidayNode) continue;

                            // Get the date and holiday name as strings
                            xmlChar* dateContent = xmlNodeGetContent(dateNode);
                            xmlChar* holidayContent = xmlNodeGetContent(holidayNode);

                            if (dateContent && holidayContent)
                            {
                                std::string date = reinterpret_cast<char *>(dateContent);
                                std::string holiday = reinterpret_cast<char *>(holidayContent);

                                // Trim whitespace
                                date.erase(0, date.find_first_not_of(" \n\r\t"));
                                date.erase(date.find_last_not_of(" \n\r\t") + 1);
                                holiday.erase(0, holiday.find_first_not_of(" \n\r\t"));
                                holiday.erase(holiday.find_last_not_of(" \n\r\t") + 1);

                                // Limit string sizes to prevent memory issues
                                if (date.length() <= 100 && holiday.length() <= 200) {
                                    bankHolidays.push_back(std::make_pair(date, holiday));
                                }
                            }

                            // Properly free the content strings
                            if (dateContent) xmlFree(dateContent);
                            if (holidayContent) xmlFree(holidayContent);
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

    // Cleanup libxml2
    xmlCleanupParser();

    return bankHolidays;
}

int main()
{
    std::string country_code;
    int year;

    std::cout << "Enter the country code (e.g., us, uk): ";
    std::cin >> country_code;
    
    // Transform to lowercase for consistency
    std::transform(country_code.begin(), country_code.end(), country_code.begin(), ::tolower);

    std::cout << "Enter the year: ";
    
    // Validate year input
    if (!(std::cin >> year)) {
        std::cerr << "Invalid year input" << std::endl;
        return 1;
    }

    // Clear any remaining input
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<std::pair<std::string, std::string>> holidays = fetchBankHolidays(country_code, year);

    if (!holidays.empty()) {
        std::cout << "| Date | Holiday |\n";
        std::cout << "|------|---------|\n";
        for (const auto &holiday : holidays)
        {
            std::cout << "| " << holiday.first << " | " << holiday.second << " |\n";
        }
    } else {
        std::cout << "No holidays found or an error occurred." << std::endl;
    }

    return 0;
}