#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

static int open_serial_device(const char *path) {
    // Open securely:
    // - O_CLOEXEC prevents FD leakage across exec (CWE-403).
    // - O_NOCTTY prevents acquiring controlling terminal.
    // - O_NOFOLLOW prevents following symlinks (CWE-59).
    // - O_NONBLOCK avoids potential blocking on open; we clear it after.
    int flags = O_RDWR | O_CLOEXEC | O_NOCTTY | O_NOFOLLOW | O_NONBLOCK;

    int fd;
    do {
        fd = open(path, flags);
    } while (fd == -1 && errno == EINTR);  // Handle EINTR (CWE-703)

    if (fd == -1) {
        return -1;
    }

    // Ensure the opened target is a character device (defense-in-depth)
    struct stat st;
    if (fstat(fd, &st) == -1) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    if (!S_ISCHR(st.st_mode)) {
        // Not a tty/char device – reject
        close(fd);
        errno = ENOTTY;
        return -1;
    }

#ifdef TIOCEXCL
    // Request exclusive access (prevents others from opening concurrently)
    if (ioctl(fd, TIOCEXCL) == -1) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
#endif

    // Restore blocking mode if desired
    int cur = fcntl(fd, F_GETFL);
    if (cur != -1) {
        (void)fcntl(fd, F_SETFL, cur & ~O_NONBLOCK);
    }

    return fd;
}

int main(void) {
    const char *dev = "/dev/ttyS0";

    int fd = open_serial_device(dev);
    if (fd == -1) {
        perror("Failed to open serial device securely");
        return 1;
    }

    // Use the device safely...
    // write(fd, ...); read(fd, ...); configure termios, etc.

    if (close(fd) == -1) {
        perror("close");
        return 1;
    }

    return 0;
}