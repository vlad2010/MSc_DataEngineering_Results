#include <stdio.h>

int main() {
    // Initialize an array of integers
    int numbers[] = {1, 2, 3, 4, 5};

    // Print the array elements
    for (int i = 0; i < sizeof(numbers) / sizeof(numbers[0]); i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    return 0;
}
