#include <stdio.h>

int main() {
    // Statically initialize an array of integers
    int numbers[] = {1, 2, 3, 4, 5};
    size_t num_elements = sizeof(numbers) / sizeof(numbers[0]);

    // Print the array elements
    for (size_t i = 0; i < num_elements; i++) {
        // Ensure format string is controlled and prevent any possible issues with unchecked format specifiers
        printf("%d ", numbers[i]);
    }
    printf("\n");

    return 0;
}