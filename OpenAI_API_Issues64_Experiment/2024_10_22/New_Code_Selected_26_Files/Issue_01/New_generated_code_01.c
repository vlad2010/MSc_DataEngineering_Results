#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int main() {
    // Open the serial port with proper error and permission handling
    int fd = open("/dev/ttyS0", O_RDWR | O_NOCTTY | O_SYNC);
    if (fd == -1) {
        // Instead of just printing "Failed to open device", include strerror and errno
        fprintf(stderr, "Error opening /dev/ttyS0: %s\n", strerror(errno));
        return 1;
    }

    // Ensure that the file descriptor is properly closed
    // This is important to prevent resource leaks
    if (close(fd) == -1) {
        // Handle the error if closing fails
        fprintf(stderr, "Error closing /dev/ttyS0: %s\n", strerror(errno));
        return 1;
    }

    return 0;
}