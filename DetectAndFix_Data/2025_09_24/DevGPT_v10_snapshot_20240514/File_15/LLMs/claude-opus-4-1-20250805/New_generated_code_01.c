#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <gcrypt.h>
#include "ssh_key_handler.h"

#define MAX_KEY_SIZE (1024 * 1024) // 1MB max key size - adjust based on requirements
#define MIN_KEY_SIZE 64            // Minimum reasonable key size

// Secure function to load an SSH private key from a file
int load_ssh_private_key(const char *filename, gcry_sexp_t *key) {
    FILE *fp = NULL;
    char *filecontent = NULL;
    int result = 0;
    
    // Input validation
    if (filename == NULL || key == NULL) {
        fprintf(stderr, "Invalid parameters\n");
        return 0;
    }
    
    // Basic path traversal check (additional validation may be needed)
    if (strstr(filename, "..") != NULL || strchr(filename, '\n') != NULL) {
        fprintf(stderr, "Invalid filename\n");
        return 0;
    }
    
    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open key file\n");
        return 0;
    }
    
    // Get file size with proper error checking
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "File seek error\n");
        fclose(fp);
        return 0;
    }
    
    long filesize = ftell(fp);
    if (filesize == -1L) {
        fprintf(stderr, "Failed to determine file size\n");
        fclose(fp);
        return 0;
    }
    
    // Validate file size
    if (filesize < MIN_KEY_SIZE || filesize > MAX_KEY_SIZE) {
        fprintf(stderr, "Invalid key file size\n");
        fclose(fp);
        return 0;
    }
    
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "File seek error\n");
        fclose(fp);
        return 0;
    }
    
    // Allocate secure memory for sensitive data
    filecontent = (char *)gcry_malloc_secure(filesize + 1);
    if (filecontent == NULL) {
        fprintf(stderr, "Failed to allocate secure memory\n");
        fclose(fp);
        return 0;
    }
    
    // Clear the allocated memory first
    memset(filecontent, 0, filesize + 1);
    
    // Read file with proper error checking
    size_t bytes_read = fread(filecontent, 1, filesize, fp);
    if (bytes_read != (size_t)filesize) {
        if (ferror(fp)) {
            fprintf(stderr, "Error reading key file\n");
        } else {
            fprintf(stderr, "Incomplete key file read\n");
        }
        gcry_free(filecontent);
        fclose(fp);
        return 0;
    }
    
    filecontent[filesize] = '\0';
    fclose(fp);
    fp = NULL;
    
    // Convert the key file content into an s-expression
    gcry_error_t err = gcry_sexp_new(key, filecontent, filesize, 0);
    
    // Securely wipe and free the memory
    gcry_free(filecontent);
    filecontent = NULL;
    
    if (err) {
        fprintf(stderr, "Failed to load private key\n");
        return 0;
    }
    
    result = 1; // Success
    return result;
}