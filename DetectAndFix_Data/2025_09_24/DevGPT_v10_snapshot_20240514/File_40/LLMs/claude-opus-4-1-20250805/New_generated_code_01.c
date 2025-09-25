#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <glob.h>
#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

#define MAX_LINE_LENGTH 4096
#define MAX_PATTERN_LENGTH 256

// Validate that the path doesn't contain directory traversal attempts
int isPathSafe(const char* path) {
    // Check for path traversal patterns
    if (strstr(path, "../") != NULL || strstr(path, "..\\") != NULL) {
        return 0;
    }
    
    // Check for absolute paths (optional - depending on requirements)
    if (path[0] == '/' || path[0] == '\\') {
        return 0;
    }
    
    // Resolve to real path and verify it's within working directory
    char resolvedPath[PATH_MAX];
    char currentDir[PATH_MAX];
    
    if (realpath(path, resolvedPath) == NULL) {
        return 0;
    }
    
    if (getcwd(currentDir, sizeof(currentDir)) == NULL) {
        return 0;
    }
    
    // Ensure resolved path is within current directory
    if (strncmp(resolvedPath, currentDir, strlen(currentDir)) != 0) {
        return 0;
    }
    
    return 1;
}

// Validate glob pattern for dangerous characters
int isPatternSafe(const char* pattern) {
    if (pattern == NULL || strlen(pattern) == 0) {
        return 0;
    }
    
    if (strlen(pattern) > MAX_PATTERN_LENGTH) {
        return 0;
    }
    
    // Check for command injection characters
    const char* dangerous = ";|&`$(){}[]<>\n\r";
    for (size_t i = 0; i < strlen(pattern); i++) {
        if (strchr(dangerous, pattern[i]) != NULL) {
            return 0;
        }
    }
    
    // Check for path traversal in pattern
    if (strstr(pattern, "../") != NULL || strstr(pattern, "..\\") != NULL) {
        return 0;
    }
    
    return 1;
}

void processFile(const char* filename) {
    // Validate the filename before processing
    if (!isPathSafe(filename)) {
        fprintf(stderr, "Security: Unsafe path detected: %s\n", filename);
        return;
    }
    
    // Check file status before opening
    struct stat fileStat;
    if (stat(filename, &fileStat) != 0) {
        fprintf(stderr, "Error: Cannot stat file: %s (errno: %d)\n", filename, errno);
        return;
    }
    
    // Verify it's a regular file
    if (!S_ISREG(fileStat.st_mode)) {
        fprintf(stderr, "Security: Not a regular file: %s\n", filename);
        return;
    }
    
    // Check file size to prevent processing huge files
    const off_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit
    if (fileStat.st_size > MAX_FILE_SIZE) {
        fprintf(stderr, "Security: File too large: %s (%ld bytes)\n", 
                filename, (long)fileStat.st_size);
        return;
    }
    
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s (errno: %d)\n", filename, errno);
        return;
    }
    
    // Use larger buffer and handle long lines properly
    char* buffer = malloc(MAX_LINE_LENGTH);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return;
    }
    
    printf("Processing file: %s\n", filename);
    
    while (fgets(buffer, MAX_LINE_LENGTH, file) != NULL) {
        // Check if line was truncated
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] != '\n' && !feof(file)) {
            fprintf(stderr, "Warning: Line truncated in file %s\n", filename);
            // Skip rest of the line
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF);
        }
        
        // Process each line safely
        // Sanitize output to prevent terminal escape sequence injection
        for (size_t i = 0; i < len; i++) {
            if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\t') {
                buffer[i] = '?';
            }
        }
        
        printf("Line: %s", buffer);
    }
    
    // Check for read errors
    if (ferror(file)) {
        fprintf(stderr, "Error reading file: %s\n", filename);
    }
    
    free(buffer);
    fclose(file);
}

void searchFiles(const char* searchPattern) {
    // Validate the search pattern
    if (!isPatternSafe(searchPattern)) {
        fprintf(stderr, "Security: Invalid or unsafe search pattern\n");
        return;
    }
    
    glob_t globResult;
    memset(&globResult, 0, sizeof(globResult));
    
    // Use GLOB_NOCHECK to handle no matches gracefully
    // Use GLOB_NOSORT for better performance if order doesn't matter
    int globFlags = GLOB_NOSORT | GLOB_NOCHECK;
    
    int result = glob(searchPattern, globFlags, NULL, &globResult);
    
    if (result != 0 && result != GLOB_NOMATCH) {
        fprintf(stderr, "Error matching files with pattern: %s (error: %d)\n", 
                searchPattern, result);
        return;
    }
    
    if (result == GLOB_NOMATCH || globResult.gl_pathc == 0) {
        printf("No files found matching pattern: %s\n", searchPattern);
        globfree(&globResult);
        return;
    }
    
    // Limit number of files to process to prevent DoS
    const size_t MAX_FILES = 1000;
    size_t filesToProcess = globResult.gl_pathc;
    
    if (filesToProcess > MAX_FILES) {
        fprintf(stderr, "Warning: Too many files (%zu), processing only first %zu\n", 
                globResult.gl_pathc, MAX_FILES);
        filesToProcess = MAX_FILES;
    }
    
    for (size_t i = 0; i < filesToProcess; i++) {
        processFile(globResult.gl_pathv[i]);
    }
    
    globfree(&globResult);
}

int main(int argc, char* argv[]) {
    // Use command line argument or default pattern
    const char* searchPattern = (argc > 1) ? argv[1] : "*.txt";
    
    // Additional validation for command line input
    if (argc > 1 && strlen(argv[1]) > MAX_PATTERN_LENGTH) {
        fprintf(stderr, "Error: Search pattern too long\n");
        return 1;
    }
    
    searchFiles(searchPattern);
    
    return 0;
}