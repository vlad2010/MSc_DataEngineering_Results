#include <stdio.h>
#include <string.h>

int main() {
    // Initialize strings with explicit null-termination checking
    const size_t buffer_size = 6;
    char str1[buffer_size] = "Hello";
    char str2[buffer_size] = "World";

    // Using strncmp instead of strcmp for added safety
    int result = strncmp(str1, str2, buffer_size - 1);

    // Output the comparison result
    printf("Output value: %d\n", result);

    return 0;
}