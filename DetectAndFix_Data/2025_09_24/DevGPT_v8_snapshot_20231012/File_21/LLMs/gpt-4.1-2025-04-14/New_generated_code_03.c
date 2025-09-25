#include <stdio.h>
#include <string.h>

#define MAX_LEN 100

int main() {
    char str1[MAX_LEN];
    char str2[MAX_LEN];

    printf("Enter first string: ");
    if (fgets(str1, sizeof(str1), stdin) == NULL) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }
    // Remove newline character if present
    str1[strcspn(str1, "\n")] = '\0';

    printf("Enter second string: ");
    if (fgets(str2, sizeof(str2), stdin) == NULL) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }
    str2[strcspn(str2, "\n")] = '\0';

    int result = strcmp(str1, str2);

    printf("Output value: %d\n", result);

    return 0;
}