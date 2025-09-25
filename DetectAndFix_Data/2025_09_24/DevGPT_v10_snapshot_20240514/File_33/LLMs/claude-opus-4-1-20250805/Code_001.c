#include <stdio.h>
#include <stdlib.h>
#include <mach-o/dyld.h>

int main() {
    char path[1024];
    uint32_t size = sizeof(path);

    // Get the path to the current executable
    if (_NSGetExecutablePath(path, &size) == 0) {
        printf("Executable path: %s\n", path);
    } else {
        printf("Buffer too small; need size %u\n", size);
    }

    return 0;
}