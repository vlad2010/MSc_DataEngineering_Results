#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glob.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctype.h>

// Security-related limits
#define MAX_FILES_TO_PROCESS 10000UL
#define MAX_BYTES_PER_FILE   (10UL * 1024UL * 1024UL) // 10 MiB cap per file
#define READ_BUF_SIZE        4096

static int is_within_base(const char *canonical, const char *base) {
    if (!canonical || !base) return 0;
    // Treat root as allowing everything ("/")
    if (strcmp(base, "/") == 0) return 1;

    size_t base_len = strlen(base);
    size_t canon_len = strlen(canonical);
    if (base_len == 0) return 0;

    // Normalize base by trimming trailing slashes (except when base == "/")
    while (base_len > 1 && base[base_len - 1] == '/') {
        base_len--;
    }

    if (canon_len < base_len) return 0;
    if (memcmp(canonical, base, base_len) != 0) return 0;

    // Must be exactly the base dir or a child (i.e., next char is '/' or end)
    if (canonical[base_len] == '\0' || canonical[base_len] == '/') return 1;
    return 0;
}

static void print_sanitized_line(const char *line, int sanitize) {
    if (!line) return;
    if (!sanitize) {
        fputs(line, stdout);
        return;
    }

    // Replace control characters except '\n' and '\t' to avoid terminal escapes
    for (const unsigned char *p = (const unsigned char *)line; *p; ++p) {
        unsigned char c = *p;
        if (c == '\n' || c == '\t') {
            fputc(c, stdout);
        } else if (isprint(c)) {
            fputc(c, stdout);
        } else {
            fputc('?', stdout);
        }
    }
}

// Safely open and process a file only if it is within allowedBaseDir,
// is not a symlink, and is a regular file.
static void processFileSafe(const char *path, const char *allowedBaseDir, int sanitizeOutput) {
    if (!path || !allowedBaseDir) return;

    // Resolve canonical path (resolves symlinks in path components)
    char *canonical = realpath(path, NULL);
    if (!canonical) {
        fprintf(stderr, "Skipping (realpath failed): %s: %s\n", path, strerror(errno));
        return;
    }

    // Enforce base directory containment
    if (!is_within_base(canonical, allowedBaseDir)) {
        fprintf(stderr, "Skipping (outside allowed base): %s\n", canonical);
        free(canonical);
        return;
    }

    // Open without following symlink on the last component
    int fd = open(canonical, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "Skipping (open failed): %s: %s\n", canonical, strerror(errno));
        free(canonical);
        return;
    }

    // Ensure it is a regular file
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Skipping (fstat failed): %s: %s\n", canonical, strerror(errno));
        close(fd);
        free(canonical);
        return;
    }
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "Skipping (not a regular file): %s\n", canonical);
        close(fd);
        free(canonical);
        return;
    }

    // Read via stdio for convenience
    FILE *file = fdopen(fd, "r");
    if (!file) {
        fprintf(stderr, "Skipping (fdopen failed): %s: %s\n", canonical, strerror(errno));
        close(fd); // fdopen failed, so fd not owned by FILE*
        free(canonical);
        return;
    }

    char buffer[READ_BUF_SIZE];
    size_t bytes_read_total = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        size_t len = strnlen(buffer, sizeof(buffer));
        bytes_read_total += len;
        if (bytes_read_total > MAX_BYTES_PER_FILE) {
            fprintf(stderr, "Truncating output (per-file byte limit reached): %s\n", canonical);
            break;
        }

        fputs("Line: ", stdout);
        print_sanitized_line(buffer, sanitizeOutput);
        // If the chunk didn't end with newline, mirror original behavior (no extra newline).
    }

    if (ferror(file)) {
        fprintf(stderr, "I/O error while reading: %s: %s\n", canonical, strerror(errno));
    }

    fclose(file); // also closes fd
    free(canonical);
}

static void searchFilesSafe(const char *searchPattern, const char *allowedBaseDir, size_t maxFiles, int sanitizeOutput) {
    if (!searchPattern || !allowedBaseDir) return;

    glob_t globResult;
    memset(&globResult, 0, sizeof(globResult));

    // Reduce CPU by avoiding sorting; do not enable tilde expansion.
    int flags = GLOB_NOSORT
#ifdef GLOB_NOESCAPE
        | GLOB_NOESCAPE
#endif
        ;

    int gret = glob(searchPattern, flags, NULL, &globResult);
    if (gret != 0) {
        if (gret == GLOB_NOMATCH) {
            fprintf(stderr, "No files matched pattern: %s\n", searchPattern);
        } else {
            fprintf(stderr, "Error matching files with pattern: %s (code=%d)\n", searchPattern, gret);
        }
        globfree(&globResult);
        return;
    }

    size_t to_process = globResult.gl_pathc;
    if (to_process > maxFiles) {
        fprintf(stderr, "Too many files matched (%zu). Limiting to first %zu to avoid DoS.\n",
                (size_t)globResult.gl_pathc, maxFiles);
        to_process = maxFiles;
    }

    for (size_t i = 0; i < to_process; i++) {
        processFileSafe(globResult.gl_pathv[i], allowedBaseDir, sanitizeOutput);
    }

    globfree(&globResult);
}

int main(void) {
    // Allowed base directory: current working directory (canonicalized)
    char *cwd = getcwd(NULL, 0);
    if (!cwd) {
        fprintf(stderr, "Failed to get current working directory: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }

    // Detect if stdout is a terminal; if so, sanitize output to avoid terminal injection.
    int sanitizeOutput = isatty(fileno(stdout)) ? 1 : 0;

    const char *searchPattern = "*.txt"; // Example wildcard search pattern

    searchFilesSafe(searchPattern, cwd, MAX_FILES_TO_PROCESS, sanitizeOutput);

    free(cwd);
    return 0;
}