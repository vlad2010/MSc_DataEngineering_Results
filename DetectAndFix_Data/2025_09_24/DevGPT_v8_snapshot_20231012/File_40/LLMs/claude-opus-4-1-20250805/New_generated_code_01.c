#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

// Safe bubble sort with input validation
int bubbleSort(int arr[], int n) {
    // Input validation
    if (arr == NULL) {
        fprintf(stderr, "Error: NULL array pointer provided\n");
        return -1;
    }
    
    if (n <= 0) {
        fprintf(stderr, "Error: Invalid array size (n=%d)\n", n);
        return -1;
    }
    
    // Prevent potential integer overflow issues
    if (n > INT_MAX - 1) {
        fprintf(stderr, "Error: Array size too large\n");
        return -1;
    }
    
    // Reasonable upper limit for array size (prevent excessive memory operations)
    const int MAX_ARRAY_SIZE = 1000000;
    if (n > MAX_ARRAY_SIZE) {
        fprintf(stderr, "Error: Array size exceeds maximum allowed (%d)\n", MAX_ARRAY_SIZE);
        return -1;
    }
    
    // Actual bubble sort algorithm
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            // Additional bounds checking (defensive programming)
            assert(j >= 0 && j < n - 1);
            
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    
    return 0; // Success
}

// Safe array printing function
void printArray(const int arr[], int n) {
    if (arr == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid array or size for printing\n");
        return;
    }
    
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    // Validate array size calculation
    if (n <= 0 || n > INT_MAX / sizeof(int)) {
        fprintf(stderr, "Error: Invalid array size calculation\n");
        return EXIT_FAILURE;
    }
    
    printf("Original array: ");
    printArray(arr, n);
    
    // Check return value of bubbleSort
    if (bubbleSort(arr, n) != 0) {
        fprintf(stderr, "Error: Sorting failed\n");
        return EXIT_FAILURE;
    }
    
    printf("Sorted array: ");
    printArray(arr, n);
    
    return EXIT_SUCCESS;
}