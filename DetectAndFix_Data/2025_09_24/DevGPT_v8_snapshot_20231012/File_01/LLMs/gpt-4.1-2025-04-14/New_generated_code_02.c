#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <errno.h>

int main(void) {
    int fd = open("/dev/ttyS0", O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (fd == -1) {
        perror("Failed to open device");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        perror("fstat failed");
        close(fd);
        return 1;
    }

    // Check if the file is a character device
    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "Error: /dev/ttyS0 is not a character device\n");
        close(fd);
        return 1;
    }

    // Device is safely opened and validated
    // ... your code here ...

    close(fd);
    return 0;
}