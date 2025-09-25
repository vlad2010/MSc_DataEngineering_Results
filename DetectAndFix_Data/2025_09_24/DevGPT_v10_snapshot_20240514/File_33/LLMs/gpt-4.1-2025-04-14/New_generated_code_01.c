#include <stdio.h>
#include <stdlib.h>
#include <mach-o/dyld.h>

int main() {
    uint32_t size = 0;
    char *path = NULL;

    // First, get the required size
    _NSGetExecutablePath(NULL, &size);

    path = malloc(size);
    if (path == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    if (_NSGetExecutablePath(path, &size) == 0) {
        // Ensure null-termination (should be, but extra safety)
        path[size - 1] = '\0';
        printf("Executable path: %s\n", path);
    } else {
        fprintf(stderr, "Failed to get executable path\n");
        free(path);
        return 1;
    }

    free(path);
    return 0;
}