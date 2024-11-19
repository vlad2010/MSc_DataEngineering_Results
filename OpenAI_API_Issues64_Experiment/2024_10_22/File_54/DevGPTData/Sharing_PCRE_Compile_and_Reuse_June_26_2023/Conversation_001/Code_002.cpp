pcre* compileRegex(const char* pattern) {
    const char* error;
    int errorOffset;
    pcre* re = pcre_compile(pattern, 0, &error, &errorOffset, NULL);
    if (re == nullptr) {
        std::cerr << "Failed to compile regex: " << error << std::endl;
    }
    return re;
}
