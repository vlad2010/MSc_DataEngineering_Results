#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <unistd.h>

int main(void) {
    uint32_t size = 0;
    char *path = NULL;
    int rc;

    // Robustly obtain the executable path without assuming a fixed-size buffer.
    // _NSGetExecutablePath returns 0 on success, -1 if the buffer is too small and sets 'size' to required length.
    // The API docs do not guarantee that passing NULL is valid, so use a grow-and-retry loop.
    uint32_t cap = 0;
    do {
        // If size is 0 (first iteration), start with a reasonable guess (e.g., 1024).
        cap = (size > 0) ? size : 1024;
        char *tmp = (char *)realloc(path, cap);
        if (!tmp) {
            perror("realloc");
            free(path);
            return 1;
        }
        path = tmp;
        size = cap;

        rc = _NSGetExecutablePath(path, &size);
        // On failure, 'size' is updated to the required size, and we loop to realloc accordingly.
    } while (rc != 0);

    // Canonicalize to eliminate symlinks/relative segments before any security-sensitive use.
    // This addresses CWE-59/CWE-180 by ensuring we operate on a resolved, absolute path.
    char *resolved = realpath(path, NULL); // POSIX: passing NULL asks realpath() to allocate
    if (!resolved) {
        // If realpath fails (unexpected for the running executable), fall back to the original.
        // Be cautious: non-canonical path should not be used for security decisions.
        fprintf(stderr, "Warning: could not canonicalize executable path (%s). Using non-canonical path.\n",
                strerror(errno));
        resolved = path;
        path = NULL; // avoid double free
    }

    // Note: Printing absolute paths may leak information (CWE-200).
    // Gate this behind a debug flag in production code.
    printf("Executable path: %s\n", resolved);

    // Cleanup
    if (resolved != NULL && resolved != path) {
        free(resolved);
    }
    free(path);

    return 0;
}