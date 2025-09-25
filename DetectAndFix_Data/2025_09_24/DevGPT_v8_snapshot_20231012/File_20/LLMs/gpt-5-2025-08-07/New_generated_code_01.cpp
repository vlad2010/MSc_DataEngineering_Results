#include <string>
#include <algorithm>
#include <cctype>

#if defined(_WIN32) || defined(_WIN64)
  #include <windows.h>
  #include <shellapi.h>
#elif defined(__APPLE__) || defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
  #include <unistd.h>
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <errno.h>
  #include <string.h>
  #include <sys/stat.h>
#endif

// Simple URL validator: allow only http/https/mailto, disallow control chars/whitespace.
// Tailor to your threat model if you need more schemes or stricter checks.
static bool is_safe_url(const std::string& url) {
    if (url.empty() || url.size() > 2048) return false;

    // Disallow leading '-' to avoid any chance of option-style confusion.
    if (url[0] == '-') return false;

    // No control or whitespace characters
    for (unsigned char ch : url) {
        if (ch <= 0x1F || ch == 0x7F || ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
            return false;
    }

    // Extract scheme
    const auto colon = url.find(':');
    if (colon == std::string::npos) return false;
    std::string scheme = url.substr(0, colon);
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c){ return (char)std::tolower(c); });

    if (scheme != "http" && scheme != "https" && scheme != "mailto") return false;

    // Require "://" for http(s)
    if ((scheme == "http" || scheme == "https")) {
        if (url.size() < colon + 3 || url[colon + 1] != '/' || url[colon + 2] != '/')
            return false;
    }

    return true;
}

#if defined(_WIN32) || defined(_WIN64)
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), (int)s.size(), nullptr, 0);
    if (len <= 0) {
        // Fallback without MB_ERR_INVALID_CHARS if needed
        len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
        if (len <= 0) throw std::runtime_error("UTF-8 to UTF-16 conversion failed");
    }
    std::wstring out((size_t)len, L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &out[0], len) != len) {
        throw std::runtime_error("UTF-8 to UTF-16 conversion failed");
    }
    return out;
}
#else
static bool is_exec_file(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    if (!S_ISREG(st.st_mode)) return false;
    return (access(path, X_OK) == 0);
}
#endif

// Returns true on success, false on failure.
bool open_url(const std::string &url) {
    if (!is_safe_url(url)) {
        // log/handle invalid input
        return false;
    }

#if defined(_WIN32) || defined(_WIN64)
    try {
        std::wstring wurl = utf8_to_wide(url);
        // Use ShellExecuteW: no shell parsing of our own, lets Windows resolve the URL via registered handlers.
        HINSTANCE h = ShellExecuteW(nullptr, L"open", wurl.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        // Per docs, values <= 32 indicate an error.
        if ((INT_PTR)h <= 32) {
            // Optionally map error codes in h to messages
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }

#elif defined(__APPLE__)
    // Prefer absolute path to avoid CWE-426.
    const char* open_path = "/usr/bin/open";
    // Fallback to PATH search only if absolute path not present.
    bool use_abs = is_exec_file(open_path);

    pid_t pid = fork();
    if (pid == 0) {
        if (use_abs) {
            execl(open_path, "open", url.c_str(), (char*)nullptr);
        } else {
            execlp("open", "open", url.c_str(), (char*)nullptr);
        }
        _exit(127); // exec failed
    } else if (pid < 0) {
        return false;
    } else {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) return false;
        if (WIFEXITED(status)) {
            return (WEXITSTATUS(status) == 0);
        }
        return false;
    }

#elif defined(__linux__)
    const char* xdg_path = "/usr/bin/xdg-open";
    bool use_abs = is_exec_file(xdg_path);

    pid_t pid = fork();
    if (pid == 0) {
        if (use_abs) {
            execl(xdg_path, "xdg-open", url.c_str(), (char*)nullptr);
        } else {
            execlp("xdg-open", "xdg-open", url.c_str(), (char*)nullptr);
        }
        _exit(127);
    } else if (pid < 0) {
        return false;
    } else {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) return false;
        if (WIFEXITED(status)) {
            return (WEXITSTATUS(status) == 0);
        }
        return false;
    }

#elif defined(__ANDROID__)
    // Using the Activity/Intent API via JNI is preferred on Android.
    // If that's not available here, we can invoke "am" safely without a shell:
    const char* am_path = "/system/bin/am";
    bool use_abs = is_exec_file(am_path);

    pid_t pid = fork();
    if (pid == 0) {
        if (use_abs) {
            execl(am_path, "am", "start", "-a", "android.intent.action.VIEW", "-d", url.c_str(), (char*)nullptr);
        } else {
            execlp("am", "am", "start", "-a", "android.intent.action.VIEW", "-d", url.c_str(), (char*)nullptr);
        }
        _exit(127);
    } else if (pid < 0) {
        return false;
    } else {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) return false;
        if (WIFEXITED(status)) {
            return (WEXITSTATUS(status) == 0);
        }
        return false;
    }

#else
    // Unsupported platform
    (void)url;
    return false;
#endif
}