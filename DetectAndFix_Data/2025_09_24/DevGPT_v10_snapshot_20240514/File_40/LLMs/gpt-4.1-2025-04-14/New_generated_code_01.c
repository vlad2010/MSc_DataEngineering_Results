#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <glob.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>

#define BASE_DIRECTORY "./" // Restrict to current directory

// Helper to check if path is within BASE_DIRECTORY
int is_path_safe(const char* base, const char* path) {
    char resolved_base[PATH_MAX];
    char resolved_path[PATH_MAX];

    if (!realpath(base, resolved_base) || !realpath(path, resolved_path)) {
        return 0;
    }
    // Ensure resolved_path starts with resolved_base
    return strncmp(resolved_path, resolved_base, strlen(resolved_base)) == 0;
}

void processFile(const char* filename) {
    struct stat st;
    if (stat(filename, &st) != 0) {
        fprintf(stderr, "Error stating file: %s (%s)\n", filename, strerror(errno));
        return;
    }
    // Only process regular files
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "Skipping non-regular file: %s\n", filename);
        return;
    }
    // Restrict to files within BASE_DIRECTORY
    if (!is_path_safe(BASE_DIRECTORY, filename)) {
        fprintf(stderr, "File outside allowed directory: %s\n", filename);
        return;
    }

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s (%s)\n", filename, strerror(errno));
        return;
    }

    char buffer[256];
    size_t line_num = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        line_num++;
        // Warn if line was truncated
        if (strchr(buffer, '\n') == NULL && !feof(file)) {
            fprintf(stderr, "Warning: Line %zu in %s is too long and was truncated.\n", line_num, filename);
            // Discard rest of the line
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF);
        }
        printf("Line: %s", buffer);
    }

    fclose(file);
}

void searchFiles(const char* searchPattern) {
    // Validate searchPattern: only allow [a-zA-Z0-9_.*?/-]
    for (const char* p = searchPattern; *p; ++p) {
        if (!(isalnum((unsigned char)*p) || strchr("._*?/-", *p))) {
            fprintf(stderr, "Invalid character in search pattern: %c\n", *p);
            return;
        }
    }

    glob_t globResult;
    if (glob(searchPattern, 0, NULL, &globResult) != 0) {
        fprintf(stderr, "Error matching files with pattern: %s\n", searchPattern);
        return;
    }

    for (size_t i = 0; i < globResult.gl_pathc; i++) {
        processFile(globResult.gl_pathv[i]);
    }

    globfree(&globResult);
}

int main(int argc, char* argv[]) {
    const char* searchPattern = "*.txt"; // Default pattern

    if (argc > 1) {
        searchPattern = argv[1];
    }

    searchFiles(searchPattern);

    return 0;
}