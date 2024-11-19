#include <iostream>
#include <string>
#include <curl/curl.h>
#include <libxml/HTMLparser.h>
#include <libxml/xpath.h>

// Rest of the code...

int main() {
    // Rest of the code...

    // Create an XPath context
    xmlXPathContextPtr xpathCtx = xmlXPathNewContext(doc);
    if (xpathCtx) {
        // Register the namespaces used in the XPath expressions
        xmlXPathRegisterNsFunc(xpathCtx, BAD_CAST "html", BAD_CAST "http://www.w3.org/1999/xhtml", xmlXPathNamespacesDefault);
        
        // Rest of the code...

        // Cleanup XPath context
        xmlXPathFreeContext(xpathCtx);
    }

    // Rest of the code...

    return 0;
}
