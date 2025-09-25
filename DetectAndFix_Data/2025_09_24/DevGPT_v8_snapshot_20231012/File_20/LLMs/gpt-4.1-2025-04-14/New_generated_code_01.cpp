#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <shellapi.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#endif

void open_url(const std::string &url) {
#if defined(_WIN32) || defined(_WIN64)
    // Use ShellExecute to open URL
    HINSTANCE result = ShellExecuteA(
        NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL
    );
    if ((INT_PTR)result <= 32) {
        // Handle error
    }
#elif defined(__APPLE__)
    pid_t pid;
    const char *argv[] = {"open", url.c_str(), NULL};
    int status;
    if (posix_spawn(&pid, "/usr/bin/open", NULL, NULL, (char * const *)argv, NULL) == 0) {
        waitpid(pid, &status, 0);
        if (status != 0) {
            // Handle error
        }
    } else {
        // Handle error
    }
#elif defined(__linux__)
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execlp("xdg-open", "xdg-open", url.c_str(), (char *)NULL);
        _exit(127); // exec failed
    } else if (pid > 0) {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (status != 0) {
            // Handle error
        }
    } else {
        // Handle fork error
    }
#else
    // Unsupported platform
#endif
}