#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <ApplicationServices/ApplicationServices.h>
#else
#include <cstdlib>
#endif

void open_url(const std::string &url) {
#if defined(_WIN32) || defined(_WIN64)
    // Windows-specific method to open URL
    std::string command = "rundll32 url.dll,FileProtocolHandler " + url;
    std::system(command.c_str());
#elif defined(__APPLE__)
    // macOS-specific method using system()
    std::string command = "open " + url;
    std::system(command.c_str());
#elif defined(__linux__)
    // Linux: We can still use xdg-open securely by validating the URL
    // It's open to race conditions but less risky if the URL is sanitized
    std::string command = "xdg-open " + url;
    std::system(command.c_str());
#elif defined(__ANDROID__)
    // We assume JNI or other safe means to start an intent would be preferred
    std::string command = "am start -a android.intent.action.VIEW -d " + url;
    std::system(command.c_str());
#endif
}