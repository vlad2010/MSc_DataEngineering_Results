xmlChar* dumpedHtml;
int dumpedLength;
htmlDocDumpMemory(doc, &dumpedHtml, &dumpedLength);
std::cout << "Parsed HTML content: " << dumpedHtml << std::endl;
xmlFree(dumpedHtml);
