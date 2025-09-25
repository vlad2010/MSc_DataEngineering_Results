#include <stdio.h>
#include <stdlib.h>
#include <gcrypt.h>
#include "ssh_key_handler.h"

// Example function to load an SSH private key from a file
// This is a simplified version and might need adjustments based on the key format and encryption requirements
int load_ssh_private_key(const char *filename, gcry_sexp_t *key) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open key file: %s\n", filename);
        return 0;
    }

    // Seek to the end of the file to determine the file size
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate memory for the file content
    char *filecontent = (char *)malloc(filesize + 1);
    if (filecontent == NULL) {
        fclose(fp);
        fprintf(stderr, "Failed to allocate memory for key file\n");
        return 0;
    }

    // Read the file into memory
    fread(filecontent, 1, filesize, fp);
    filecontent[filesize] = '\0'; // Null-terminate the string

    fclose(fp);

    // Convert the key file content into a s-expression
    gcry_error_t err = gcry_sexp_new(key, filecontent, filesize, 0);
    free(filecontent);

    if (err) {
        fprintf(stderr, "Failed to load private key: %s\n", gcry_strerror(err));
        return 0;
    }

    return 1; // Success
}