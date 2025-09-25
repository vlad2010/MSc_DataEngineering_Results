#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <glob.h>

void processFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Read and process the file contents here
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        // Process each line of the file here
        printf("Line: %s", buffer);
    }
    
    fclose(file);
}

void searchFiles(const char* searchPattern) {
    glob_t globResult;
    
    if (glob(searchPattern, 0, NULL, &globResult) != 0) {
        printf("Error matching files with pattern: %s\n", searchPattern);
        return;
    }
    
    for (size_t i = 0; i < globResult.gl_pathc; i++) {
        processFile(globResult.gl_pathv[i]);
    }
    
    globfree(&globResult);
}

int main() {
    const char* searchPattern = "*.txt"; // Example wildcard search pattern

    searchFiles(searchPattern);

    return 0;
}