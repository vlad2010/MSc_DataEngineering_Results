#include <string>
#include <regex>
#include <stdexcept>
#include <cstdlib>
#include <array>
#include <memory>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <shellapi.h>
#elif defined(__APPLE__) || defined(__linux__)
    #include <unistd.h>
    #include <sys/wait.h>
#endif

void open_url(const std::string &url) {
    // Validate URL format - only allow http/https URLs with safe characters
    // This regex allows standard URL characters and prevents injection
    std::regex url_pattern(
        R"(^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$)",
        std::regex::ECMAScript
    );
    
    if (!std::regex_match(url, url_pattern)) {
        throw std::invalid_argument("Invalid URL format or contains unsafe characters");
    }
    
    // Additional length check to prevent buffer overflows
    if (url.length() > 2048) {
        throw std::invalid_argument("URL too long");
    }

#if defined(_WIN32) || defined(_WIN64)
    // Use Windows API instead of system() for safety
    HINSTANCE result = ShellExecuteA(
        nullptr,
        "open",
        url.c_str(),
        nullptr,
        nullptr,
        SW_SHOWNORMAL
    );
    
    // ShellExecute returns > 32 on success
    if (reinterpret_cast<intptr_t>(result) <= 32) {
        throw std::runtime_error("Failed to open URL on Windows");
    }
    
#elif defined(__APPLE__) || defined(__linux__)
    // Use fork/exec instead of system() to avoid shell interpretation
    pid_t pid = fork();
    
    if (pid == -1) {
        throw std::runtime_error("Fork failed");
    } else if (pid == 0) {
        // Child process
        #if defined(__linux__)
            const char* cmd = "xdg-open";
        #elif defined(__APPLE__)
            const char* cmd = "open";
        #endif
        
        // Execute with URL as separate argument (not interpreted by shell)
        execl("/usr/bin/env", "env", cmd, url.c_str(), nullptr);
        
        // If execl returns, it failed
        _exit(1);
    } else {
        // Parent process - wait for child
        int status;
        waitpid(pid, &status, 0);
        
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Failed to open URL");
        }
    }
    
#elif defined(__ANDROID__)
    // For Android, use JNI to call Intent properly instead of shell command
    // This is a placeholder - actual implementation would require JNI
    throw std::runtime_error("Android URL opening requires JNI implementation");
    
#else
    throw std::runtime_error("Unsupported platform");
#endif
}

// Alternative safer implementation using only execvp (POSIX systems)
#if !defined(_WIN32) && !defined(_WIN64)
void open_url_safe_posix(const std::string &url) {
    // Validate URL as before
    std::regex url_pattern(
        R"(^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$)",
        std::regex::ECMAScript
    );
    
    if (!std::regex_match(url, url_pattern) || url.length() > 2048) {
        throw std::invalid_argument("Invalid URL");
    }
    
    pid_t pid = fork();
    if (pid == -1) {
        throw std::runtime_error("Fork failed");
    } else if (pid == 0) {
        #if defined(__linux__)
            const char* argv[] = {"xdg-open", url.c_str(), nullptr};
            execvp("xdg-open", const_cast<char**>(argv));
        #elif defined(__APPLE__)
            const char* argv[] = {"open", url.c_str(), nullptr};
            execvp("open", const_cast<char**>(argv));
        #endif
        _exit(1);
    } else {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Failed to open URL");
        }
    }
}
#endif