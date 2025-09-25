#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

int countingSort(int arr[], int n) {
    // Input validation
    if (arr == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return -1;
    }
    
    // Find the minimum and maximum elements
    int min = arr[0];
    int max = arr[0];
    
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    
    // Check for integer overflow in range calculation
    if (min < 0 && max > INT_MAX + min) {
        fprintf(stderr, "Error: Range too large, would cause integer overflow\n");
        return -1;
    }
    
    // Calculate range safely
    long long range = (long long)max - (long long)min + 1;
    
    // Check if range is reasonable for memory allocation
    if (range > INT_MAX / sizeof(int) || range > 100000000) { // 100MB limit
        fprintf(stderr, "Error: Range too large for memory allocation\n");
        return -1;
    }
    
    // Allocate counting array with proper size
    size_t count_size = (size_t)range * sizeof(int);
    int* count = (int*)calloc(range, sizeof(int));
    
    if (count == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for counting array\n");
        return -1;
    }
    
    // Count the occurrences of each element (adjusted for negative values)
    for (int i = 0; i < n; i++) {
        int index = arr[i] - min;
        // Additional bounds check for safety
        if (index < 0 || index >= range) {
            free(count);
            fprintf(stderr, "Error: Index out of bounds\n");
            return -1;
        }
        count[index]++;
    }
    
    // Modify the counting array to store the position of each element
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }
    
    // Allocate output array
    int* output = (int*)malloc(n * sizeof(int));
    if (output == NULL) {
        free(count);
        fprintf(stderr, "Error: Memory allocation failed for output array\n");
        return -1;
    }
    
    // Build the output array by placing elements in their correct positions
    for (int i = n - 1; i >= 0; i--) {
        int index = arr[i] - min;
        // Additional bounds check
        if (index < 0 || index >= range || count[index] <= 0 || count[index] > n) {
            free(count);
            free(output);
            fprintf(stderr, "Error: Invalid count array state\n");
            return -1;
        }
        output[count[index] - 1] = arr[i];
        count[index]--;
    }
    
    // Copy the sorted elements back to the original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
    
    // Free allocated memory
    free(count);
    free(output);
    
    return 0;
}

int main() {
    int arr[] = {4, 2, 10, 8, 7, 1, 5, 6, 3, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    
    if (countingSort(arr, n) == 0) {
        printf("\nSorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    } else {
        printf("\nSorting failed due to error\n");
    }
    
    // Test with negative numbers
    int arr2[] = {-5, 3, -2, 0, 7, -1};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);
    
    printf("\nOriginal array with negatives: ");
    for (int i = 0; i < n2; i++) {
        printf("%d ", arr2[i]);
    }
    
    if (countingSort(arr2, n2) == 0) {
        printf("\nSorted array: ");
        for (int i = 0; i < n2; i++) {
            printf("%d ", arr2[i]);
        }
        printf("\n");
    } else {
        printf("\nSorting failed due to error\n");
    }
    
    return 0;
}