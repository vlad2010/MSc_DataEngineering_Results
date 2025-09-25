#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach-o/dyld.h>
#include <limits.h>

int main() {
    char *path = NULL;
    uint32_t size = 0;
    int result = -1;
    
    // First call to get the required buffer size
    _NSGetExecutablePath(NULL, &size);
    
    // Validate size is reasonable (prevent integer overflow and excessive allocation)
    if (size == 0 || size > PATH_MAX * 4) {
        fprintf(stderr, "Error: Invalid buffer size returned: %u\n", size);
        return 1;
    }
    
    // Dynamically allocate buffer with the exact required size
    path = (char *)malloc(size);
    if (path == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for path buffer\n");
        return 1;
    }
    
    // Clear the buffer to ensure null-termination
    memset(path, 0, size);
    
    // Get the executable path with properly sized buffer
    result = _NSGetExecutablePath(path, &size);
    if (result == 0) {
        // Ensure null-termination (defensive programming)
        path[size - 1] = '\0';
        
        // Use %s format specifier safely - path is treated as string, not format
        printf("Executable path: %s\n", path);
    } else {
        // This shouldn't happen if we sized the buffer correctly
        fprintf(stderr, "Error: Failed to get executable path even with sized buffer\n");
        free(path);
        return 1;
    }
    
    // Clean up allocated memory
    free(path);
    return 0;
}