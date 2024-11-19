xpathObj = xmlXPathEvalExpression(BAD_CAST "//table[@class='country-table']//tr[position() > 1]", xpathCtx);
if (xpathObj == NULL) {
    std::cerr << "XPath evaluation failed." << std::endl;
    // Handle the error
}
