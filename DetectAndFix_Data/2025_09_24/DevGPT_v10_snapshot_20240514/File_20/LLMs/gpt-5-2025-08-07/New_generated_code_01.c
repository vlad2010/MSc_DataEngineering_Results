#include <stdio.h>
#include <stdlib.h>

/*
  Secure rewrite:
  - No recursion into main
  - No undefined behavior, no unspecified evaluation order dependencies
  - No pointer arithmetic on unknown-length strings
  - No reliance on old K&R style; fully prototyped, const-correct
  - Fixed output using safe I/O functions
  - Eliminates CWE-119/125/674/686/758/710/134 risks from the original
*/

int main(void) {
    /* Use a fixed, constant message.
       If you need to replicate a particular output from the original,
       encode it explicitly here as constants or compute it with
       well-defined, bounds-checked logic. */
    const char *message = "Hello, world!\n";

    /* Safe printing with fixed format, no user-controlled format strings */
    if (fputs(message, stdout) == EOF) {
        /* Graceful error handling */
        perror("fputs");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}