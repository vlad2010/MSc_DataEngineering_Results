#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>

#define FILE_PATH "/sys/class/backlight/intel_backlight/brightness"
#define MAX_BRIGHTNESS 120000
#define MIN_BRIGHTNESS 3000
#define BRIGHTNESS_INCREMENT 12000

static void die_errno(const char* msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static void die_msg(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static int parse_int_strict(const char* s, int* out) {
    // Trim leading whitespace
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || *s == '\f' || *s == '\v') {
        s++;
    }
    if (*s == '\0') {
        return 0;
    }

    errno = 0;
    char* end = NULL;
    long val = strtol(s, &end, 10);
    if (errno == ERANGE || end == s) {
        return 0;
    }

    // Allow trailing newline/whitespace only
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r' || *end == '\f' || *end == '\v') {
        end++;
    }
    if (*end != '\0') {
        return 0;
    }

    if (val < INT_MIN || val > INT_MAX) {
        return 0;
    }
    *out = (int)val;
    return 1;
}

static int read_brightness(const char* path) {
    // O_NOFOLLOW mitigates symlink tricks on the final path component (CWE-367)
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd == -1) {
        die_errno("open (read)");
    }

    // sysfs values are short; 64 bytes is ample
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n < 0) {
        int e = errno;
        close(fd);
        errno = e;
        die_errno("read");
    }
    buf[n] = '\0';

    if (close(fd) == -1) {
        die_errno("close (read)");
    }

    int value = 0;
    if (!parse_int_strict(buf, &value)) {
        die_msg("invalid numeric contents in brightness file");
    }

    return value;
}

static void write_brightness(const char* path, int brightness) {
    int fd = open(path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd == -1) {
        die_errno("open (write)");
    }

    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%d", brightness);
    if (len < 0 || len >= (int)sizeof(buf)) {
        int e = errno;
        close(fd);
        errno = e ? e : EOVERFLOW;
        die_errno("snprintf");
    }

    ssize_t written = write(fd, buf, (size_t)len);
    if (written < 0 || written != len) {
        int e = errno;
        close(fd);
        errno = (written < 0) ? e : EIO;
        die_errno("write");
    }

    // Ensure the write is flushed to the backing store (for sysfs this may be a no-op, but harmless)
    if (fsync(fd) == -1) {
        int e = errno;
        close(fd);
        errno = e;
        die_errno("fsync");
    }

    if (close(fd) == -1) {
        die_errno("close (write)");
    }
}

static long clamp_long(long v, long lo, long hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <up|down>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int dir = 0; // +1 for up, -1 for down
    if (strcmp(argv[1], "up") == 0) {
        dir = +1;
        printf("Brightness up\n");
    } else if (strcmp(argv[1], "down") == 0) {
        dir = -1;
        printf("Brightness down\n");
    } else {
        fprintf(stderr, "Unknown option: %s\nUsage: %s <up|down>\n", argv[1], argv[0]);
        return EXIT_FAILURE;
    }

    int current_brightness = read_brightness(FILE_PATH);

    // Use long to avoid any potential overflow in arithmetic (defensive)
    long new_brightness_long = (long)current_brightness + (long)dir * (long)BRIGHTNESS_INCREMENT;
    new_brightness_long = clamp_long(new_brightness_long, MIN_BRIGHTNESS, MAX_BRIGHTNESS);
    int new_brightness = (int)new_brightness_long;

    printf("Current brightness: %d\n", current_brightness);
    printf("New brightness: %d\n", new_brightness);

    write_brightness(FILE_PATH, new_brightness);

    return EXIT_SUCCESS;
}