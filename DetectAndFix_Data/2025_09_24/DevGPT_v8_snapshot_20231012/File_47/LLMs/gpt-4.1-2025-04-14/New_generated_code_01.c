#include <stdio.h>
#include <stddef.h> // for size_t

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

void print_array(const int *numbers, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");
}

int main() {
    // Initialize an array of integers
    int numbers[] = {1, 2, 3, 4, 5};

    // Print the array elements
    print_array(numbers, ARRAY_SIZE(numbers));

    return 0;
}