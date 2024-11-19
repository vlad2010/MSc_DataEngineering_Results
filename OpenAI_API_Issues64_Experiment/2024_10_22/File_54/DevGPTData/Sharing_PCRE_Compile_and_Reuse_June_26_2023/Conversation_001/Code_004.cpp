int main() {
    const char* regexPattern = "hello\\s+world";
    const char* subject = "hello   world";

    pcre* compiledRegex = compileRegex(regexPattern);
    if (compiledRegex != nullptr) {
        matchRegex(compiledRegex, subject);
        pcre_free(compiledRegex);  // Free the compiled pattern after you're done with it
    }

    return 0;
}
