#include <iostream>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <libxml/HTMLparser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

// Callback function to write fetched data into a string
size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *response)
{
    size_t total_size = size * nmemb;
    response->append(static_cast<char *>(contents), total_size);
    return total_size;
}

std::vector<std::pair<std::string, std::string>> fetchBankHolidays(std::string country, int year)
{
    std::vector<std::pair<std::string, std::string>> bankHolidays;

    // Create the URL string
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
            htmlDocPtr doc = htmlReadDoc(reinterpret_cast<const xmlChar *>(response.c_str()), nullptr, nullptr, HTML_PARSE_NOERROR | HTML_PARSE_NOWARNING);
            if (doc)
            {
                // Create an XPath context
                xmlXPathContextPtr xpathCtx = xmlXPathNewContext(doc);
                if (xpathCtx)
                {
                    // Register the namespaces used in the XPath expressions
                    xmlXPathRegisterNs(xpathCtx, BAD_CAST "html", BAD_CAST "http://www.w3.org/1999/xhtml");

                    // Evaluate the XPath expression to select the table rows
                    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(BAD_CAST "//html:table[@id='public_holidays']/html:tbody/html:tr", xpathCtx);
                    if (xpathObj)
                    {
                        // Iterate over the selected table rows
                        xmlNodeSetPtr nodes = xpathObj->nodesetval;
                        if (nodes != NULL)
                        {
                            for (int i = 0; i < nodes->nodeNr; ++i)
                            {
                                // Extract the date and holiday name from each row
                                xmlNodePtr node = nodes->nodeTab[i];
                                xmlNodePtr dateNode = xmlFirstElementChild(node);
                                xmlNodePtr holidayNode = xmlNextElementSibling(dateNode);

                                // Get the date and holiday name as strings
                                std::string date = reinterpret_cast<char *>(xmlNodeGetContent(dateNode));
                                std::string holiday = reinterpret_cast<char *>(xmlNodeGetContent(holidayNode));

                                // Add the date and holiday name to the vector
                                bankHolidays.push_back(std::make_pair(date, holiday));

                                // Cleanup the content strings
                                xmlFree(dateNode);
                                xmlFree(holidayNode);
                            }
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

    return bankHolidays;
}

int main()
{
    std::string country_code;
    int year;

    std::cout << "Enter the country code (e.g., us, uk): ";
    std::cin >> country_code;
    std::cout << "Enter the year: ";
    std::cin >> year;

    std::vector<std::pair<std::string, std::string>> holidays = fetchBankHolidays(country_code, year);

    std::cout << "| Date | Holiday |\n";
    std::cout << "|------|---------|\n";
    for (const auto &holiday : holidays)
    {
        std::cout << "| " << holiday.first << " | " << holiday.second << " |\n";
    }

    return 0;
}
