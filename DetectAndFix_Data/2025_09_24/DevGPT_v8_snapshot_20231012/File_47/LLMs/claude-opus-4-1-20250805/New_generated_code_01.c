#include <stdio.h>
#include <stddef.h>

int main(void) {
    // Initialize an array of integers (const for read-only data)
    const int numbers[] = {1, 2, 3, 4, 5};
    
    // Calculate array size once (more efficient for larger loops)
    const size_t array_size = sizeof(numbers) / sizeof(numbers[0]);
    
    // Print the array elements using size_t for indexing
    for (size_t i = 0; i < array_size; i++) {
        // Using specific format specifier and limiting output
        printf("%d ", numbers[i]);
    }
    printf("\n");
    
    // Explicit flush to ensure output is written
    fflush(stdout);
    
    return 0;
}