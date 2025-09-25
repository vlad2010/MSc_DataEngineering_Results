#include <stdio.h>
#include <stdlib.h>
#include <gcrypt.h>
#include <errno.h>
#include <limits.h>
#include "ssh_key_handler.h"

// Example function to load an SSH private key from a file
int load_ssh_private_key(const char *filename, gcry_sexp_t *key) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open key file\n"); // Avoid leaking filename
        return 0;
    }

    // Seek to the end of the file to determine the file size
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "Failed to seek to end of key file\n");
        fclose(fp);
        return 0;
    }
    long filesize = ftell(fp);
    if (filesize < 0 || filesize > INT_MAX) {
        fprintf(stderr, "Invalid key file size\n");
        fclose(fp);
        return 0;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to start of key file\n");
        fclose(fp);
        return 0;
    }

    // Allocate memory for the file content
    char *filecontent = (char *)malloc((size_t)filesize + 1);
    if (filecontent == NULL) {
        fclose(fp);
        fprintf(stderr, "Failed to allocate memory for key file\n");
        return 0;
    }

    // Read the file into memory
    size_t read_bytes = fread(filecontent, 1, (size_t)filesize, fp);
    if (read_bytes != (size_t)filesize) {
        fprintf(stderr, "Failed to read complete key file\n");
        free(filecontent);
        fclose(fp);
        return 0;
    }
    filecontent[filesize] = '\0'; // Null-terminate for safety, though not used by gcry_sexp_new

    fclose(fp);

    // Convert the key file content into a s-expression
    gcry_error_t err = gcry_sexp_new(key, filecontent, (size_t)filesize, 0);
    free(filecontent);

    if (err) {
        fprintf(stderr, "Failed to load private key: %s\n", gcry_strerror(err));
        return 0;
    }

    return 1; // Success
}