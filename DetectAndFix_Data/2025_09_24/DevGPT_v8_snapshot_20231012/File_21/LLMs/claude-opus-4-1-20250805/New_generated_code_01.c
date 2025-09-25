#include <stdio.h>
#include <string.h>

int main(void) {
    // Using const for immutable strings (best practice)
    const char str1[] = "Hello";
    const char str2[] = "World";

    // strcmp returns <0 if str1<str2, 0 if equal, >0 if str1>str2
    int result = strcmp(str1, str2);

    // More descriptive output
    printf("Comparison result: %d\n", result);
    
    if (result < 0) {
        printf("\"%s\" comes before \"%s\" lexicographically\n", str1, str2);
    } else if (result > 0) {
        printf("\"%s\" comes after \"%s\" lexicographically\n", str1, str2);
    } else {
        printf("\"%s\" and \"%s\" are equal\n", str1, str2);
    }

    return 0;
}