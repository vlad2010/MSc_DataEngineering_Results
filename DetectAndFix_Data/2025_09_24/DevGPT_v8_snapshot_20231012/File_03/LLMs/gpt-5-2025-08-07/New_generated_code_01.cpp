#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <curl/curl.h>
#include <libxml/HTMLparser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

struct ResponseBuffer {
    std::string* out;
    size_t max_bytes;
};

// Safe write callback with size limit
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    ResponseBuffer* buf = static_cast<ResponseBuffer*>(userp);
    if (!buf || !buf->out) return 0;

    if (buf->out->size() + total_size > buf->max_bytes) {
        // Abort transfer if exceeding limit
        return 0; // causes CURLE_WRITE_ERROR
    }
    buf->out->append(static_cast<const char*>(contents), total_size);
    return total_size;
}

static bool isValidCountryCode(const std::string& country) {
    // Accept 2-32 lowercase letters and optional hyphens (defense-in-depth).
    // Adjust as needed for your use-case.
    static const std::regex re("^[a-z]([a-z-]{0,30}[a-z])?$");
    return std::regex_match(country, re);
}

std::vector<std::pair<std::string, std::string>> fetchBankHolidays(const std::string& country, int year)
{
    std::vector<std::pair<std::string, std::string>> bankHolidays;

    // Validate inputs early
    if (!isValidCountryCode(country)) {
        std::cerr << "Invalid country code format.\n";
        return bankHolidays;
    }
    if (year < 1900 || year > 2100) {
        std::cerr << "Year out of accepted range (1900-2100).\n";
        return bankHolidays;
    }

    std::string url = "https://www.officeholidays.com/countries/" + country + "/" + std::to_string(year);

    // Initialize curl globally (thread-safe init for whole process).
    static bool curl_initialized = false;
    if (!curl_initialized) {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
            std::cerr << "curl_global_init() failed\n";
            return bankHolidays;
        }
        curl_initialized = true;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl_easy_init() failed\n";
        return bankHolidays;
    }

    char errbuf[CURL_ERROR_SIZE] = {0};

    // Response buffer with limit (e.g., 5 MB)
    std::string response;
    ResponseBuffer respBuf{ &response, 5 * 1024 * 1024 };

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &respBuf);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);

    // Timeouts (DoS mitigation)
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    // TLS: be explicit
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    // Restrict protocols
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTPS);
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTPS);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 3L);

    // Reasonable UA
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "holiday-fetcher/1.0");

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        const char* msg = errbuf[0] ? errbuf : curl_easy_strerror(res);
        std::cerr << "curl_easy_perform() failed: " << msg << std::endl;
        curl_easy_cleanup(curl);
        return bankHolidays;
    }

    curl_easy_cleanup(curl);

    // Parse HTML safely: disable any network access in the parser
    // Note: HTML_PARSE_NONET blocks network, NOERROR/NOWARNING keeps logs clean
    htmlDocPtr doc = htmlReadDoc(
        reinterpret_cast<const xmlChar *>(response.c_str()),
        nullptr,
        nullptr,
        HTML_PARSE_NOERROR | HTML_PARSE_NOWARNING | HTML_PARSE_NONET
    );

    if (!doc) {
        std::cerr << "Failed to parse HTML document.\n";
        return bankHolidays;
    }

    xmlXPathContextPtr xpathCtx = xmlXPathNewContext(doc);
    if (!xpathCtx) {
        std::cerr << "Failed to create XPath context.\n";
        xmlFreeDoc(doc);
        return bankHolidays;
    }

    // Register XHTML namespace used in XPath
    if (xmlXPathRegisterNs(xpathCtx, BAD_CAST "html", BAD_CAST "http://www.w3.org/1999/xhtml") != 0) {
        // Continue anyway; some pages may not require ns
    }

    xmlXPathObjectPtr xpathObj = xmlXPathEvalExpression(
        BAD_CAST "//html:table[@id='public_holidays']/html:tbody/html:tr", xpathCtx);

    if (!xpathObj) {
        std::cerr << "XPath evaluation failed.\n";
        xmlXPathFreeContext(xpathCtx);
        xmlFreeDoc(doc);
        return bankHolidays;
    }

    xmlNodeSetPtr nodes = xpathObj->nodesetval;
    if (nodes && nodes->nodeNr > 0) {
        for (int i = 0; i < nodes->nodeNr; ++i) {
            xmlNodePtr row = nodes->nodeTab[i];
            if (!row) continue;

            xmlNodePtr dateNode = xmlFirstElementChild(row);
            if (!dateNode) continue;

            xmlNodePtr holidayNode = xmlNextElementSibling(dateNode);
            if (!holidayNode) continue;

            xmlChar* dateContent = xmlNodeGetContent(dateNode);
            xmlChar* holidayContent = xmlNodeGetContent(holidayNode);
            if (!dateContent || !holidayContent) {
                if (dateContent) xmlFree(dateContent);
                if (holidayContent) xmlFree(holidayContent);
                continue;
            }

            std::string date(reinterpret_cast<char*>(dateContent));
            std::string holiday(reinterpret_cast<char*>(holidayContent));
            bankHolidays.emplace_back(date, holiday);

            // Free the content buffers returned by xmlNodeGetContent
            xmlFree(dateContent);
            xmlFree(holidayContent);
        }
    }

    xmlXPathFreeObject(xpathObj);
    xmlXPathFreeContext(xpathCtx);
    xmlFreeDoc(doc);

    return bankHolidays;
}

int main()
{
    std::string country_code;
    int year;

    std::cout << "Enter the country code (e.g., us, uk): ";
    if (!(std::cin >> country_code)) {
        std::cerr << "Failed to read country code.\n";
        return 1;
    }

    // Normalize to lowercase
    for (char& c : country_code) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    std::cout << "Enter the year: ";
    if (!(std::cin >> year)) {
        std::cerr << "Failed to read year.\n";
        return 1;
    }

    if (!isValidCountryCode(country_code)) {
        std::cerr << "Invalid country code format (only lowercase letters and hyphens allowed, 2-32 chars).\n";
        return 1;
    }
    if (year < 1900 || year > 2100) {
        std::cerr << "Year out of accepted range (1900-2100).\n";
        return 1;
    }

    auto holidays = fetchBankHolidays(country_code, year);

    std::cout << "| Date | Holiday |\n";
    std::cout << "|------|---------|\n";
    for (const auto &holiday : holidays) {
        std::cout << "| " << holiday.first << " | " << holiday.second << " |\n";
    }

    return 0;
}