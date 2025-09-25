#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

#define FILE_PATH "/sys/class/backlight/intel_backlight/brightness"
#define MAX_BRIGHTNESS 120000
#define MIN_BRIGHTNESS 3000
#define BRIGHTNESS_INCREMENT 12000
#define MAX_STR_LENGTH 20

void handle_error(const char* message) {
    fprintf(stderr, "Error: %s: %s\n", message, strerror(errno));
    exit(EXIT_FAILURE);
}

int safe_atoi(const char* str, int* result) {
    char* endptr;
    long val;
    
    errno = 0;
    val = strtol(str, &endptr, 10);
    
    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) {
        return -1;  // Overflow/underflow
    }
    
    if (endptr == str || *endptr != '\0' && *endptr != '\n') {
        return -1;  // No conversion or invalid characters
    }
    
    *result = (int)val;
    return 0;
}

int read_brightness(const char* file_path) {
    // Open with O_NOFOLLOW to prevent symlink attacks
    int fd = open(file_path, O_RDONLY | O_NOFOLLOW);
    if (fd == -1) {
        handle_error("open for reading");
    }
    
    // Acquire shared lock for reading
    if (flock(fd, LOCK_SH) == -1) {
        close(fd);
        handle_error("flock for reading");
    }
    
    // Verify it's a regular file in /sys
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        flock(fd, LOCK_UN);
        close(fd);
        handle_error("fstat");
    }
    
    // Check if it's in the expected /sys filesystem
    if (!S_ISREG(sb.st_mode)) {
        flock(fd, LOCK_UN);
        close(fd);
        fprintf(stderr, "Error: Not a regular file\n");
        exit(EXIT_FAILURE);
    }
    
    char file_contents[MAX_STR_LENGTH];
    ssize_t bytes_read = read(fd, file_contents, MAX_STR_LENGTH - 1);
    if (bytes_read == -1) {
        flock(fd, LOCK_UN);
        close(fd);
        handle_error("read");
    }
    
    file_contents[bytes_read] = '\0';
    
    // Release lock and close
    flock(fd, LOCK_UN);
    close(fd);
    
    int brightness;
    if (safe_atoi(file_contents, &brightness) != 0) {
        fprintf(stderr, "Error: Invalid brightness value in file\n");
        exit(EXIT_FAILURE);
    }
    
    return brightness;
}

void write_brightness(const char* file_path, int brightness) {
    // Open with O_NOFOLLOW to prevent symlink attacks
    int fd = open(file_path, O_WRONLY | O_NOFOLLOW);
    if (fd == -1) {
        handle_error("open for writing");
    }
    
    // Acquire exclusive lock for writing
    if (flock(fd, LOCK_EX) == -1) {
        close(fd);
        handle_error("flock for writing");
    }
    
    // Verify it's still the expected file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        flock(fd, LOCK_UN);
        close(fd);
        handle_error("fstat");
    }
    
    if (!S_ISREG(sb.st_mode)) {
        flock(fd, LOCK_UN);
        close(fd);
        fprintf(stderr, "Error: Not a regular file\n");
        exit(EXIT_FAILURE);
    }
    
    char buffer[MAX_STR_LENGTH];
    int len = snprintf(buffer, sizeof(buffer), "%d\n", brightness);
    if (len < 0 || len >= sizeof(buffer)) {
        flock(fd, LOCK_UN);
        close(fd);
        fprintf(stderr, "Error: Failed to format brightness value\n");
        exit(EXIT_FAILURE);
    }
    
    ssize_t bytes_written = write(fd, buffer, len);
    if (bytes_written != len) {
        flock(fd, LOCK_UN);
        close(fd);
        handle_error("write");
    }
    
    // Ensure data is written to disk
    if (fsync(fd) == -1) {
        flock(fd, LOCK_UN);
        close(fd);
        handle_error("fsync");
    }
    
    // Release lock and close
    flock(fd, LOCK_UN);
    if (close(fd) == -1) {
        handle_error("close");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <up|down>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // Drop privileges if running as SUID (security best practice)
    uid_t real_uid = getuid();
    uid_t effective_uid = geteuid();
    if (real_uid != effective_uid) {
        // Running as SUID, drop privileges after file operations
        if (setuid(real_uid) != 0) {
            handle_error("setuid");
        }
    }
    
    int current_brightness = read_brightness(FILE_PATH);
    int new_brightness;
    
    if (strcmp(argv[1], "up") == 0) {
        printf("Brightness up\n");
        // Check for integer overflow
        if (current_brightness > INT_MAX - BRIGHTNESS_INCREMENT) {
            new_brightness = MAX_BRIGHTNESS;
        } else {
            new_brightness = current_brightness + BRIGHTNESS_INCREMENT;
        }
    } else if (strcmp(argv[1], "down") == 0) {
        printf("Brightness down\n");
        // Check for integer underflow
        if (current_brightness < INT_MIN + BRIGHTNESS_INCREMENT) {
            new_brightness = MIN_BRIGHTNESS;
        } else {
            new_brightness = current_brightness - BRIGHTNESS_INCREMENT;
        }
    } else {
        fprintf(stderr, "Unknown option: %s\n", argv[1]);
        fprintf(stderr, "Usage: %s <up|down>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // Enforce brightness limits
    if (new_brightness > MAX_BRIGHTNESS) {
        new_brightness = MAX_BRIGHTNESS;
    } else if (new_brightness < MIN_BRIGHTNESS) {
        new_brightness = MIN_BRIGHTNESS;
    }
    
    printf("Current brightness: %d\n", current_brightness);
    printf("New brightness: %d\n", new_brightness);
    
    write_brightness(FILE_PATH, new_brightness);
    
    return EXIT_SUCCESS;
}