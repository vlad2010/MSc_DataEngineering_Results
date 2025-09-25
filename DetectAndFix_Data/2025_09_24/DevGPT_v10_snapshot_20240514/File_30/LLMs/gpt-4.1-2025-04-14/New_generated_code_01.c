#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#define FILE_PATH "/sys/class/backlight/intel_backlight/brightness"
#define MAX_BRIGHTNESS 120000
#define MIN_BRIGHTNESS 3000
#define BRIGHTNESS_INCREMENT 12000
#define MAX_STR_LENGTH 20

void handle_error(const char* message) {
    perror(message);
    exit(EXIT_FAILURE);
}

// Secure string to int conversion with error checking
int safe_atoi(const char* str, int* out) {
    char* endptr;
    long val;

    errno = 0;
    val = strtol(str, &endptr, 10);

    if (errno != 0 || endptr == str || *endptr != '\n' && *endptr != '\0') {
        return -1; // Conversion error
    }
    if (val < INT_MIN || val > INT_MAX) {
        return -1; // Out of int range
    }
    *out = (int)val;
    return 0;
}

int read_brightness(const char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        handle_error("fopen");
    }

    char file_contents[MAX_STR_LENGTH];
    if (fgets(file_contents, sizeof(file_contents), file) == NULL) {
        fclose(file);
        handle_error("fgets");
    }
    fclose(file);

    int brightness;
    if (safe_atoi(file_contents, &brightness) != 0) {
        fprintf(stderr, "Invalid brightness value in file.\n");
        exit(EXIT_FAILURE);
    }
    return brightness;
}

void write_brightness(const char* file_path, int brightness) {
    FILE* file = fopen(file_path, "w");
    if (!file) {
        handle_error("fopen");
    }

    if (fprintf(file, "%d", brightness) < 0) {
        fclose(file);
        handle_error("fprintf");
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <up|down>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int current_brightness = read_brightness(FILE_PATH);
    int new_brightness = current_brightness;

    if (strcmp(argv[1], "up") == 0) {
        printf("Brightness up\n");
        new_brightness = current_brightness + BRIGHTNESS_INCREMENT;
    } else if (strcmp(argv[1], "down") == 0) {
        printf("Brightness down\n");
        new_brightness = current_brightness - BRIGHTNESS_INCREMENT;
    } else {
        // Avoid format string vulnerability by using "%s"
        fprintf(stderr, "Unknown option: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

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